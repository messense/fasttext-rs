// Dictionary: tokenization, vocabulary, subwords, word n-grams, hash table

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use crate::args::{Args, ModelName};
use crate::error::{FastTextError, Result};
use crate::model::MinstdRng;
use crate::utils::{self, hash};

/// EOS (end-of-sentence) token string.
pub const EOS: &str = "</s>";
/// Beginning-of-word marker for subword computation.
pub const BOW: &str = "<";
/// End-of-word marker for subword computation.
pub const EOW: &str = ">";

/// Maximum vocabulary size for the open-addressing hash table.
pub const MAX_VOCAB_SIZE: usize = 30_000_000;
/// Maximum tokens per line.
pub const MAX_LINE_SIZE: usize = 1024;

/// Word entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum EntryType {
    /// Regular word.
    Word = 0,
    /// Label (starts with label prefix).
    Label = 1,
}

/// A vocabulary entry.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The word string.
    pub word: String,
    /// Occurrence count.
    pub count: i64,
    /// Entry type (word or label).
    pub entry_type: EntryType,
    /// Subword IDs (character n-grams + word itself). Filled by dictionary-subwords feature.
    pub subwords: Vec<i32>,
}

/// Dictionary holding vocabulary, tokenization logic, and open-addressing hash table.
///
/// The hash table (word2int) maps from FNV-1a hash position to word index in `words`.
/// It uses linear probing: slot = hash(w) % len, then (slot+1) % len, etc.
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Args reference (shared with caller).
    args: Arc<Args>,
    /// Open-addressing hash table: index → word index (-1 if empty).
    word2int: Vec<i32>,
    /// Vocabulary entries in order.
    words: Vec<Entry>,
    /// Subsampling discard probability table.
    pdiscard: Vec<f32>,
    /// Total number of entries (words + labels).
    size: i32,
    /// Number of word entries.
    nwords: i32,
    /// Number of label entries.
    nlabels: i32,
    /// Total number of tokens seen during vocabulary building.
    ntokens: i64,
    /// Prune index size (-1 = no pruning, 0+ = pruned).
    pruneidx_size: i64,
    /// Prune index mapping (for quantization).
    pruneidx: HashMap<i32, i32>,
}

impl Dictionary {
    /// Create a new empty Dictionary with the standard MAX_VOCAB_SIZE hash table.
    pub fn new(args: Arc<Args>) -> Self {
        Dictionary::new_with_capacity(args, MAX_VOCAB_SIZE)
    }

    /// Create a new empty Dictionary with a custom hash table capacity.
    ///
    /// The capacity should be larger than the expected vocabulary to keep load factor low.
    /// In production, use `new()` which uses MAX_VOCAB_SIZE (30M).
    /// For testing with small vocabularies, a smaller capacity (e.g., 1024) is sufficient.
    pub fn new_with_capacity(args: Arc<Args>, capacity: usize) -> Self {
        Dictionary {
            word2int: vec![-1i32; capacity],
            words: Vec::new(),
            pdiscard: Vec::new(),
            size: 0,
            nwords: 0,
            nlabels: 0,
            ntokens: 0,
            pruneidx_size: -1,
            pruneidx: HashMap::new(),
            args,
        }
    }

    // -------------------------------------------------------------------------
    // Hash table internals
    // -------------------------------------------------------------------------

    /// Find the hash table slot for a word (computes hash internally).
    fn find_slot(&self, w: &str) -> usize {
        self.find_slot_with_hash(w, hash(w.as_bytes()))
    }

    /// Find the hash table slot for a word given its precomputed hash.
    ///
    /// Returns the slot index where either:
    /// - `word2int[slot] == -1` (empty slot, word not in table), or
    /// - `words[word2int[slot]].word == w` (found the word).
    pub fn find_slot_with_hash(&self, w: &str, h: u32) -> usize {
        let len = self.word2int.len();
        let mut id = (h as usize) % len;
        while self.word2int[id] != -1 && self.words[self.word2int[id] as usize].word != w {
            id = (id + 1) % len;
        }
        id
    }

    // -------------------------------------------------------------------------
    // Vocabulary management
    // -------------------------------------------------------------------------

    /// Add a token to the vocabulary (or increment its count if already present).
    ///
    /// Increments `ntokens` for every call.
    pub fn add(&mut self, w: &str) {
        let h = self.find_slot(w);
        self.ntokens += 1;
        if self.word2int[h] == -1 {
            let entry_type = self.get_type_from_str(w);
            let entry = Entry {
                word: w.to_string(),
                count: 1,
                entry_type,
                subwords: Vec::new(),
            };
            self.words.push(entry);
            self.word2int[h] = self.size;
            self.size += 1;
        } else {
            self.words[self.word2int[h] as usize].count += 1;
        }
    }

    /// Get the vocabulary index of a word, or -1 if not found (OOV).
    pub fn get_id(&self, w: &str) -> i32 {
        let h = self.find_slot(w);
        self.word2int[h]
    }

    /// Get the vocabulary index of a word given its hash, or -1 if not found.
    pub fn get_id_with_hash(&self, w: &str, h: u32) -> i32 {
        let idx = self.find_slot_with_hash(w, h);
        self.word2int[idx]
    }

    /// Get the entry type (Word or Label) for a word by its vocabulary index.
    pub fn get_type_by_id(&self, id: i32) -> EntryType {
        debug_assert!(id >= 0 && (id as usize) < self.words.len());
        self.words[id as usize].entry_type
    }

    /// Get the entry type for a word string by checking the label prefix.
    ///
    /// A word is a label iff it starts with `args.label()` (default: `"__label__"`).
    pub fn get_type_from_str(&self, w: &str) -> EntryType {
        if w.starts_with(self.args.label()) {
            EntryType::Label
        } else {
            EntryType::Word
        }
    }

    /// Get the word string for a given vocabulary index.
    pub fn get_word(&self, id: i32) -> &str {
        debug_assert!(id >= 0 && (id as usize) < self.words.len());
        &self.words[id as usize].word
    }

    /// Get the label string for a given label index (0-based within labels).
    pub fn get_label(&self, lid: i32) -> Result<&str> {
        if lid < 0 || lid >= self.nlabels {
            return Err(FastTextError::InvalidArgument(format!(
                "Label id {} is out of range [0, {}]",
                lid, self.nlabels
            )));
        }
        Ok(&self.words[(lid + self.nwords) as usize].word)
    }

    // -------------------------------------------------------------------------
    // Count accessors
    // -------------------------------------------------------------------------

    /// Number of word entries (non-label).
    pub fn nwords(&self) -> i32 {
        self.nwords
    }

    /// Number of label entries.
    pub fn nlabels(&self) -> i32 {
        self.nlabels
    }

    /// Total number of tokens seen during vocabulary building.
    pub fn ntokens(&self) -> i64 {
        self.ntokens
    }

    /// Total number of vocabulary entries (words + labels).
    pub fn size(&self) -> i32 {
        self.size
    }

    // -------------------------------------------------------------------------
    // Sorting and thresholding
    // -------------------------------------------------------------------------

    /// Filter vocabulary by minimum count and rebuild the hash table.
    ///
    /// After this call:
    /// - Words with count < `t` are removed.
    /// - Labels with count < `tl` are removed.
    /// - Remaining entries sorted: words before labels, descending count within each type.
    /// - Hash table rebuilt with the same capacity.
    pub fn threshold(&mut self, t: i64, tl: i64) {
        // Sort: words (type 0) before labels (type 1), descending count within type.
        self.words.sort_unstable_by(|a, b| {
            if a.entry_type != b.entry_type {
                (a.entry_type as i8).cmp(&(b.entry_type as i8))
            } else {
                b.count.cmp(&a.count)
            }
        });

        // Remove entries below threshold.
        self.words.retain(|e| match e.entry_type {
            EntryType::Word => e.count >= t,
            EntryType::Label => e.count >= tl,
        });
        self.words.shrink_to_fit();

        // Rebuild hash table (keep same capacity, fill with -1, re-insert).
        self.size = 0;
        self.nwords = 0;
        self.nlabels = 0;
        self.word2int.fill(-1);

        for i in 0..self.words.len() {
            let h = self.find_slot(&self.words[i].word.clone());
            self.word2int[h] = i as i32;
            self.size += 1;
            match self.words[i].entry_type {
                EntryType::Word => self.nwords += 1,
                EntryType::Label => self.nlabels += 1,
            }
        }
    }

    // -------------------------------------------------------------------------
    // Tokenization
    // -------------------------------------------------------------------------

    /// Tokenize a text string into tokens, following C++ `readWord` semantics.
    ///
    /// Rules:
    /// - Splits on ASCII whitespace: space, tab (`\t`), vertical tab (`\v`/`\x0b`),
    ///   form feed (`\f`/`\x0c`), null (`\0`), carriage return (`\r`), newline (`\n`).
    /// - Newlines produce an EOS token (`"</s>"`).
    /// - Consecutive non-newline whitespace is collapsed (skipped).
    /// - UTF-8 multi-byte sequences are preserved intact (bytes processed as-is).
    /// - At most `MAX_LINE_SIZE` (1024) non-EOS tokens per line. Excess silently dropped.
    /// - A "line" is the sequence of tokens between EOS tokens (or start/end of input).
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut word_bytes: Vec<u8> = Vec::new();
        let mut line_tokens: usize = 0;

        while i < bytes.len() {
            let c = bytes[i];
            let is_ws = matches!(c, b' ' | b'\t' | b'\x0b' | b'\x0c' | b'\0' | b'\r' | b'\n');

            if is_ws {
                if word_bytes.is_empty() {
                    if c == b'\n' {
                        // Empty buffer + newline → EOS token; reset line counter.
                        tokens.push(EOS.to_string());
                        line_tokens = 0;
                    }
                    // Non-newline whitespace with empty buffer: skip.
                    i += 1;
                } else {
                    // End of current word token.
                    if line_tokens < MAX_LINE_SIZE {
                        // SAFETY: word_bytes contains valid bytes from the UTF-8 input
                        // (we push raw bytes directly), so from_utf8_lossy is safe.
                        let word = String::from_utf8_lossy(&word_bytes).into_owned();
                        tokens.push(word);
                        line_tokens += 1;
                    }
                    word_bytes.clear();

                    if c == b'\n' {
                        // Don't advance i: next iteration processes '\n' with empty buffer → EOS.
                    } else {
                        i += 1;
                    }
                }
            } else {
                word_bytes.push(c);
                i += 1;
            }
        }

        // Flush any remaining word (no trailing newline).
        if !word_bytes.is_empty() && line_tokens < MAX_LINE_SIZE {
            let word = String::from_utf8_lossy(&word_bytes).into_owned();
            tokens.push(word);
        }

        tokens
    }

    // -------------------------------------------------------------------------
    // Stream-based word reader (for readFromFile)
    // -------------------------------------------------------------------------

    /// Read one word from a reader, following C++ `readWord` semantics.
    ///
    /// Returns `true` if a token was produced (written to `word`), `false` at EOF
    /// with no remaining data. Uses `pending_newline` to simulate the C++ "ungetc"
    /// behavior when a word is terminated by a newline character.
    ///
    /// The `pending_newline` flag should be initialized to `false` before the first call
    /// and passed through subsequent calls on the same reader.
    pub fn read_word_from_reader<R: Read>(
        reader: &mut R,
        pending_newline: &mut bool,
        word: &mut String,
    ) -> bool {
        word.clear();

        // If a newline was "put back" by the previous call, produce EOS now.
        if *pending_newline {
            *pending_newline = false;
            word.push_str(EOS);
            return true;
        }

        let mut buf = [0u8; 1];
        // Accumulate raw bytes so that multi-byte UTF-8 sequences are preserved intact.
        // Converting individual bytes via `c as char` would corrupt any byte > 127,
        // turning multi-byte code points into garbage Latin-1 characters.
        let mut word_bytes: Vec<u8> = Vec::new();
        loop {
            match reader.read(&mut buf) {
                Ok(0) => {
                    // EOF: flush accumulated bytes if any.
                    if !word_bytes.is_empty() {
                        *word = String::from_utf8_lossy(&word_bytes).into_owned();
                        return true;
                    }
                    return false;
                }
                Ok(_) => {
                    let c = buf[0];
                    let is_ws =
                        matches!(c, b' ' | b'\n' | b'\r' | b'\t' | b'\x0b' | b'\x0c' | b'\0');

                    if is_ws {
                        if word_bytes.is_empty() {
                            if c == b'\n' {
                                word.push_str(EOS);
                                return true;
                            }
                            // Skip non-newline whitespace when buffer is empty.
                        } else {
                            // Token complete: convert accumulated bytes to String.
                            *word = String::from_utf8_lossy(&word_bytes).into_owned();
                            if c == b'\n' {
                                // Put back the newline via the pending flag.
                                *pending_newline = true;
                            }
                            return true;
                        }
                    } else {
                        word_bytes.push(c);
                    }
                }
                Err(_) => {
                    if !word_bytes.is_empty() {
                        *word = String::from_utf8_lossy(&word_bytes).into_owned();
                        return true;
                    }
                    return false;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Vocabulary building from file
    // -------------------------------------------------------------------------

    /// Build vocabulary by reading tokens from a reader.
    ///
    /// Reads until EOF, incrementally thresholding if the vocabulary grows too large
    /// (> 75% of table capacity). After reading, applies final threshold using
    /// `args.min_count()` and `args.min_count_label()`, then computes the discard table.
    ///
    /// Note: `init_ngrams()` (subwords) is called separately in the dictionary-subwords feature.
    pub fn read_from_file<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        let mut word = String::new();
        let mut pending_newline = false;
        let mut min_threshold: i64 = 1;
        let capacity = self.word2int.len() as i32;

        loop {
            let found = Self::read_word_from_reader(reader, &mut pending_newline, &mut word);
            if !found {
                break;
            }
            self.add(&word);

            if self.size > (0.75 * capacity as f64) as i32 {
                min_threshold += 1;
                self.threshold(min_threshold, min_threshold);
            }
        }

        self.threshold(
            self.args.min_count() as i64,
            self.args.min_count_label() as i64,
        );
        self.init_table_discard();
        self.init_ngrams();

        if self.size == 0 {
            return Err(FastTextError::InvalidArgument(
                "Empty vocabulary. Try a smaller -minCount value.".to_string(),
            ));
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Discard probability table
    // -------------------------------------------------------------------------

    /// Initialize the subsampling discard probability table.
    ///
    /// `pdiscard[i] = sqrt(t / f) + t / f` where `f = count / ntokens`.
    fn init_table_discard(&mut self) {
        self.pdiscard.resize(self.size as usize, 0.0);
        for i in 0..self.size as usize {
            let f = self.words[i].count as f32 / self.ntokens as f32;
            self.pdiscard[i] = (self.args.t() as f32 / f).sqrt() + self.args.t() as f32 / f;
        }
    }

    /// Call `init_table_discard()` publicly (used after binary load).
    pub fn init_discard(&mut self) {
        self.init_table_discard();
    }

    // -------------------------------------------------------------------------
    // Subword (n-gram) computation
    // -------------------------------------------------------------------------

    /// Push a subword hash into the hashes vector.
    ///
    /// No-op when:
    /// - `bucket == 0` (subword computation disabled)
    /// - `pruneidx_size == 0` (all subwords pruned)
    /// - `id < 0`
    ///
    /// When `pruneidx_size > 0`, maps `id` through `pruneidx` (drops if not found).
    /// Otherwise (pruneidx_size < 0, normal operation), pushes `nwords + id`.
    fn push_hash(&self, hashes: &mut Vec<i32>, id: i32) {
        if self.args.bucket() == 0 || self.pruneidx_size == 0 || id < 0 {
            return;
        }
        if self.pruneidx_size > 0 {
            if let Some(&mapped) = self.pruneidx.get(&id) {
                hashes.push(self.nwords + mapped);
            }
            // If id not in pruneidx, silently drop it.
            return;
        }
        hashes.push(self.nwords + id);
    }

    /// Compute character n-gram subwords for a word (already with BOW/EOW markers).
    ///
    /// Extracts all n-grams of length [minn, maxn] (counted in Unicode characters).
    /// UTF-8 continuation bytes (0x80–0xBF) are skipped for the outer position loop
    /// but included within n-gram bytes.
    ///
    /// Each n-gram is hashed via FNV-1a and mapped to `hash % bucket`. The resulting
    /// subword ID (`nwords + (hash % bucket)`) is pushed via `push_hash`.
    ///
    /// No-op if `maxn == 0` or `bucket == 0`.
    pub fn compute_subwords(&self, word: &str, ngrams: &mut Vec<i32>) {
        let minn = self.args.minn();
        let maxn = self.args.maxn();
        let bucket = self.args.bucket();

        if maxn == 0 || bucket == 0 {
            return;
        }

        let bytes = word.as_bytes();
        let n_bytes = bytes.len();

        let mut i = 0usize;
        while i < n_bytes {
            // Skip UTF-8 continuation bytes (0x80–0xBF) for the outer position.
            if (bytes[i] & 0xC0) == 0x80 {
                i += 1;
                continue;
            }

            let mut ngram: Vec<u8> = Vec::new();
            let mut j = i;
            let mut n: i32 = 1;

            while j < n_bytes && n <= maxn {
                // Consume the leading byte of the current Unicode character.
                ngram.push(bytes[j]);
                j += 1;
                // Consume any UTF-8 continuation bytes.
                while j < n_bytes && (bytes[j] & 0xC0) == 0x80 {
                    ngram.push(bytes[j]);
                    j += 1;
                }

                // Include this n-gram if length meets minn and is not a BOW/EOW singleton.
                // The exclusion condition skips:
                //   - The BOW '<' alone (n==1 && i==0)
                //   - The EOW '>' alone (n==1 && j==n_bytes, meaning we consumed the last char)
                if n >= minn && !(n == 1 && (i == 0 || j == n_bytes)) {
                    let h = (crate::utils::hash(&ngram) % bucket as u32) as i32;
                    self.push_hash(ngrams, h);
                }

                n += 1;
            }

            i += 1;
        }
    }

    /// Initialize the subword n-gram vectors for all vocabulary entries.
    ///
    /// For each word:
    /// - Clears the subwords vec.
    /// - Pushes the word's own vocabulary ID as the first element.
    /// - If word is not EOS, computes character n-grams from BOW+word+EOW and appends.
    ///
    /// Should be called after `threshold()` and `init_table_discard()`.
    pub fn init_ngrams(&mut self) {
        // Collect BOW+word+EOW strings ahead of time to avoid borrow conflicts.
        let size = self.size as usize;
        let words_with_markers: Vec<Option<String>> = (0..size)
            .map(|i| {
                if self.words[i].word != EOS {
                    Some(format!("{}{}{}", BOW, &self.words[i].word, EOW))
                } else {
                    None
                }
            })
            .collect();

        for (i, maybe_marked) in words_with_markers.into_iter().enumerate() {
            let mut subwords = vec![i as i32];
            if let Some(ref word_with_markers) = maybe_marked {
                self.compute_subwords(word_with_markers, &mut subwords);
            }
            self.words[i].subwords = subwords;
        }
    }

    /// Initialize both the discard table and n-gram subwords (used after binary load).
    pub fn init(&mut self) {
        self.init_table_discard();
        self.init_ngrams();
    }

    // -------------------------------------------------------------------------
    // Word n-gram hashing
    // -------------------------------------------------------------------------

    /// Add word n-gram hashes to `line` using a rolling hash with multiplier 116049371.
    ///
    /// For each pair of positions (i, j) with i < j < i + n, computes:
    ///   `h = h * 116049371 + hashes[j]` (64-bit arithmetic, sign-extended from i32)
    ///
    /// and pushes `h % bucket` as a subword ID via `push_hash`.
    ///
    /// For `word_ngrams == 1`, the inner loop never runs (no bigrams or larger).
    ///
    /// `hashes` contains the FNV-1a hash of each word token (as `i32`), in order.
    pub fn add_word_ngrams(&self, line: &mut Vec<i32>, hashes: &[i32], n: i32) {
        let bucket = self.args.bucket() as u64;
        if bucket == 0 || n <= 1 {
            return;
        }
        let n_hashes = hashes.len();
        for i in 0..n_hashes {
            // Sign-extend from i32 to u64 to match C++ `uint64_t h = hashes[i]`
            // where hashes[i] is int32_t (sign extension in C++).
            let mut h: u64 = hashes[i] as i64 as u64;
            let limit = n_hashes.min(i + n as usize);
            for &hj in &hashes[(i + 1)..limit] {
                h = h.wrapping_mul(116049371u64).wrapping_add(hj as i64 as u64);
                self.push_hash(line, (h % bucket) as i32);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Subword collection for a token (for getLine)
    // -------------------------------------------------------------------------

    /// Add subword IDs for a token to `line`.
    ///
    /// - If `wid >= 0` (in-vocab):
    ///   - `maxn <= 0`: push only the word ID.
    ///   - `maxn > 0`: push all subwords (word ID + n-grams).
    /// - If `wid < 0` (OOV):
    ///   - If token is not EOS and maxn > 0: compute subwords on-the-fly (no word ID).
    ///   - Otherwise: nothing pushed.
    pub fn add_subwords(&self, line: &mut Vec<i32>, token: &str, wid: i32) {
        if wid < 0 {
            // OOV: compute subwords on-the-fly if subwords are enabled.
            if token != EOS {
                let word_with_markers = format!("{}{}{}", BOW, token, EOW);
                self.compute_subwords(&word_with_markers, line);
            }
        } else if self.args.maxn() <= 0 {
            // In-vocab without subwords: just the word ID.
            line.push(wid);
        } else {
            // In-vocab with subwords: push all (word ID is first in subwords vec).
            let ngrams = &self.words[wid as usize].subwords;
            line.extend_from_slice(ngrams);
        }
    }

    /// Get the subword IDs (word ID + n-grams) for an in-vocabulary word.
    pub fn get_subwords(&self, id: i32) -> &Vec<i32> {
        debug_assert!(id >= 0 && (id as usize) < self.words.len());
        &self.words[id as usize].subwords
    }

    /// Get the subword n-gram IDs for any word string (in-vocab or OOV).
    ///
    /// - In-vocab: returns the stored subwords vec (word ID + n-grams).
    /// - OOV: computes n-grams on-the-fly (no word ID prepended).
    /// - EOS: returns empty vec if OOV.
    pub fn get_subwords_for_string(&self, word: &str) -> Vec<i32> {
        let id = self.get_id(word);
        if id >= 0 {
            return self.words[id as usize].subwords.clone();
        }
        let mut ngrams = Vec::new();
        if word != EOS {
            let word_with_markers = format!("{}{}{}", BOW, word, EOW);
            self.compute_subwords(&word_with_markers, &mut ngrams);
        }
        ngrams
    }

    /// Get n-gram subwords with their string representations.
    ///
    /// Returns a vec of `(id, ngram_string)` pairs:
    /// - If word is in vocab: first entry is `(word_id, word_string)`
    /// - Remaining entries are `(bucket_id, ngram_string)` for each character n-gram
    ///
    /// This mirrors the C++ `Dictionary::getSubwords(word, ngrams, substrings)` behavior
    /// and is used by the `print-ngrams` CLI command.
    pub fn get_ngram_strings(&self, word: &str) -> Vec<(i32, String)> {
        let mut result = Vec::new();
        let id = self.get_id(word);
        if id >= 0 {
            result.push((id, self.words[id as usize].word.clone()));
        }
        if word != EOS {
            let word_with_markers = format!("{}{}{}", BOW, word, EOW);
            self.compute_subwords_with_strings(&word_with_markers, &mut result);
        }
        result
    }

    /// Compute character n-gram subwords with their string representations.
    ///
    /// Similar to `compute_subwords` but also records each n-gram string alongside its ID.
    /// Only adds an entry when the n-gram's bucket ID is valid (not pruned).
    fn compute_subwords_with_strings(&self, word: &str, result: &mut Vec<(i32, String)>) {
        let minn = self.args.minn();
        let maxn = self.args.maxn();
        let bucket = self.args.bucket();

        if maxn == 0 || bucket == 0 {
            return;
        }

        let bytes = word.as_bytes();
        let n_bytes = bytes.len();

        let mut i = 0usize;
        while i < n_bytes {
            if (bytes[i] & 0xC0) == 0x80 {
                i += 1;
                continue;
            }

            let mut ngram: Vec<u8> = Vec::new();
            let mut j = i;
            let mut n: i32 = 1;

            while j < n_bytes && n <= maxn {
                ngram.push(bytes[j]);
                j += 1;
                while j < n_bytes && (bytes[j] & 0xC0) == 0x80 {
                    ngram.push(bytes[j]);
                    j += 1;
                }

                if n >= minn && !(n == 1 && (i == 0 || j == n_bytes)) {
                    let h = (crate::utils::hash(&ngram) % bucket as u32) as i32;
                    // Replicate push_hash logic but also capture the ngram string.
                    if self.pruneidx_size != 0 && h >= 0 {
                        let bucket_id = if self.pruneidx_size > 0 {
                            self.pruneidx.get(&h).map(|&mapped| self.nwords + mapped)
                        } else {
                            // pruneidx_size < 0 means no pruning
                            Some(self.nwords + h)
                        };
                        if let Some(id) = bucket_id {
                            let ngram_str =
                                String::from_utf8(ngram.clone()).unwrap_or_default();
                            result.push((id, ngram_str));
                        }
                    }
                }

                n += 1;
            }

            i += 1;
        }
    }

    // -------------------------------------------------------------------------
    // getLine: read one line from a reader, separate words and labels
    // -------------------------------------------------------------------------

    /// Read one "line" (terminated by EOS or EOF) from a reader.
    ///
    /// Fills `words` with word subword IDs and `labels` with label IDs (0-based,
    /// relative to `nwords`). Appends word n-grams after processing all tokens.
    ///
    /// Out-of-vocabulary words:
    /// - If `maxn > 0`: compute subwords on-the-fly and add to `words`.
    /// - If `maxn == 0`: skip entirely.
    ///
    /// Out-of-vocabulary labels are silently dropped.
    ///
    /// Returns the number of tokens read (including EOS if encountered).
    ///
    /// The `pending_newline` flag carries newline state across calls (same semantics
    /// as `read_word_from_reader`).
    pub fn get_line<R: Read>(
        &self,
        reader: &mut R,
        words: &mut Vec<i32>,
        labels: &mut Vec<i32>,
        pending_newline: &mut bool,
    ) -> i32 {
        let mut word_hashes: Vec<i32> = Vec::new();
        let mut token = String::new();
        let mut ntokens: i32 = 0;

        words.clear();
        labels.clear();

        loop {
            if !Self::read_word_from_reader(reader, pending_newline, &mut token) {
                break;
            }

            let h = crate::utils::hash(token.as_bytes());
            let wid = self.get_id_with_hash(&token, h);

            let entry_type = if wid < 0 {
                self.get_type_from_str(&token)
            } else {
                self.get_type_by_id(wid)
            };

            ntokens += 1;
            if entry_type == EntryType::Word {
                self.add_subwords(words, &token, wid);
                word_hashes.push(h as i32);
            } else if entry_type == EntryType::Label && wid >= 0 {
                labels.push(wid - self.nwords);
            }

            if token == EOS {
                break;
            }
        }

        self.add_word_ngrams(words, &word_hashes, self.args.word_ngrams());
        ntokens
    }

    /// Read one line from a string slice (no EOS separator), filling `words` and `labels`.
    ///
    /// Equivalent to C++ `getStringNoNewline`. Whitespace-separated tokens are
    /// processed the same as `get_line` but newlines do NOT produce EOS tokens —
    /// they are treated as plain whitespace.
    ///
    /// Returns the number of tokens read.
    pub fn get_line_from_str(
        &self,
        input: &str,
        words: &mut Vec<i32>,
        labels: &mut Vec<i32>,
    ) -> i32 {
        let mut word_hashes: Vec<i32> = Vec::new();
        let mut ntokens: i32 = 0;

        words.clear();
        labels.clear();

        // Split on all whitespace (including newlines — no EOS generation here).
        for token in input.split(|c: char| c.is_ascii_whitespace()) {
            if token.is_empty() {
                continue;
            }

            let h = crate::utils::hash(token.as_bytes());
            let wid = self.get_id_with_hash(token, h);

            let entry_type = if wid < 0 {
                self.get_type_from_str(token)
            } else {
                self.get_type_by_id(wid)
            };

            ntokens += 1;
            if entry_type == EntryType::Word {
                self.add_subwords(words, token, wid);
                word_hashes.push(h as i32);
            } else if entry_type == EntryType::Label && wid >= 0 {
                labels.push(wid - self.nwords);
            }

            if token == EOS {
                break;
            }
        }

        self.add_word_ngrams(words, &word_hashes, self.args.word_ngrams());
        ntokens
    }

    /// Get the discard probability for a vocabulary entry.
    pub fn get_discard(&self, id: i32) -> f32 {
        self.pdiscard[id as usize]
    }

    /// Check whether a word (by ID) should be discarded during unsupervised training.
    ///
    /// Always returns `false` for supervised models.
    pub fn discard(&self, id: i32, rand: f32) -> bool {
        debug_assert!(id >= 0 && id < self.nwords);
        if self.args.model() == ModelName::SUP {
            return false;
        }
        rand > self.pdiscard[id as usize]
    }

    // -------------------------------------------------------------------------
    // Unsupervised getLine (for CBOW / skip-gram training)
    // -------------------------------------------------------------------------

    /// Read one line from a reader for unsupervised (CBOW / skip-gram) training.
    ///
    /// This mirrors C++ `Dictionary::getLine(istream&, vector<int32_t>&, minstd_rand&)`.
    ///
    /// Key differences from the supervised `get_line`:
    /// - Only **word IDs** are added to `words` (no subwords, no labels).
    /// - **Subsampling** is applied: a word is discarded with probability
    ///   controlled by the discard table (`pdiscard`), using a random float
    ///   drawn from `rng`.
    /// - Out-of-vocabulary tokens are skipped (do not count toward `ntokens`).
    ///
    /// Returns the number of in-vocabulary tokens encountered (counting EOS
    /// as a token when it appears).
    pub fn get_line_unsupervised<R: Read>(
        &self,
        reader: &mut R,
        words: &mut Vec<i32>,
        pending_newline: &mut bool,
        rng: &mut MinstdRng,
    ) -> i32 {
        words.clear();
        let mut ntokens: i32 = 0;
        let mut token = String::new();

        loop {
            if !Self::read_word_from_reader(reader, pending_newline, &mut token) {
                break;
            }

            let h = utils::hash(token.as_bytes());
            let wid = self.get_id_with_hash(&token, h);

            if wid < 0 {
                // OOV token: skip entirely (does not count toward ntokens)
                continue;
            }

            ntokens += 1;
            if self.get_type_by_id(wid) == EntryType::Word {
                // Generate uniform random float in (0, 1) using minstd_rand
                let rand_val = rng.generate() as f32 / MinstdRng::M as f32;
                if !self.discard(wid, rand_val) {
                    words.push(wid); // word ID only, subwords retrieved later
                }
            }

            if ntokens as usize > MAX_LINE_SIZE || token == EOS {
                break;
            }
        }

        ntokens
    }

    // -------------------------------------------------------------------------
    // Accessors for external use
    // -------------------------------------------------------------------------

    /// Get the entries slice (in sorted order after `threshold()`).
    pub fn words(&self) -> &[Entry] {
        &self.words
    }

    /// Get a mutable reference to the entries (for binary loading).
    pub fn words_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.words
    }

    /// Get a reference to the args.
    pub fn args(&self) -> &Arc<Args> {
        &self.args
    }

    /// Get counts for a given entry type.
    pub fn get_counts(&self, entry_type: EntryType) -> Vec<i64> {
        self.words
            .iter()
            .filter(|e| e.entry_type == entry_type)
            .map(|e| e.count)
            .collect()
    }

    /// Get the pruneidx_size (for binary I/O).
    pub fn pruneidx_size(&self) -> i64 {
        self.pruneidx_size
    }

    /// Get the pruneidx (for binary I/O).
    pub fn pruneidx(&self) -> &HashMap<i32, i32> {
        &self.pruneidx
    }

    /// Set pruneidx from loaded binary data.
    pub fn set_pruneidx(&mut self, idx: HashMap<i32, i32>) {
        self.pruneidx_size = idx.len() as i64;
        self.pruneidx = idx;
    }

    /// Set pruneidx_size directly (for binary I/O loading, including negative sentinel).
    pub fn set_pruneidx_size(&mut self, size: i64) {
        self.pruneidx_size = size;
    }

    /// Set ntokens directly (for binary loading).
    pub fn set_ntokens(&mut self, ntokens: i64) {
        self.ntokens = ntokens;
    }

    /// Set size/nwords/nlabels directly (for binary loading).
    pub fn set_counts(&mut self, size: i32, nwords: i32, nlabels: i32) {
        self.size = size;
        self.nwords = nwords;
        self.nlabels = nlabels;
    }

    /// Push a raw entry (for binary loading).
    pub fn push_entry(&mut self, entry: Entry) {
        self.words.push(entry);
    }

    /// Rebuild the word2int hash table after binary loading.
    ///
    /// Resizes the hash table to `ceil(size / 0.7)` (matching C++ `load()` behavior)
    /// and reinserts all current entries.
    pub fn rebuild_word2int_after_load(&mut self) {
        let word2int_size = ((self.size as f64 / 0.7).ceil() as usize).max(1);
        self.word2int = vec![-1i32; word2int_size];
        for i in 0..self.size as usize {
            let w = self.words[i].word.clone();
            let h = hash(w.as_bytes());
            let slot = self.find_slot_with_hash(&w, h);
            self.word2int[slot] = i as i32;
        }
    }

    // -------------------------------------------------------------------------
    // Binary I/O
    // -------------------------------------------------------------------------

    /// Save the dictionary to binary format.
    ///
    /// Format:
    /// - size: i32
    /// - nwords: i32
    /// - nlabels: i32
    /// - ntokens: i64
    /// - pruneidx_size: i64
    /// - For each entry: null-terminated word string + count(i64) + type(i8)
    /// - If pruneidx_size > 0: that many (i32, i32) pairs
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        utils::write_i32(writer, self.size)?;
        utils::write_i32(writer, self.nwords)?;
        utils::write_i32(writer, self.nlabels)?;
        utils::write_i64(writer, self.ntokens)?;
        utils::write_i64(writer, self.pruneidx_size)?;

        for entry in &self.words {
            // Write null-terminated word string
            writer.write_all(entry.word.as_bytes())?;
            writer.write_all(&[0u8])?; // null terminator
            utils::write_i64(writer, entry.count)?;
            // Write entry type as i8 (1 byte)
            writer.write_all(&[entry.entry_type as i8 as u8])?;
        }

        // Write pruneidx pairs if pruneidx_size > 0
        if self.pruneidx_size > 0 {
            // Sort by key for deterministic output (C++ iterates unordered_map arbitrarily)
            let mut pairs: Vec<(i32, i32)> = self.pruneidx.iter().map(|(&k, &v)| (k, v)).collect();
            pairs.sort_unstable_by_key(|&(k, _)| k);
            for (first, second) in pairs {
                utils::write_i32(writer, first)?;
                utils::write_i32(writer, second)?;
            }
        }

        Ok(())
    }

    /// Load a Dictionary from binary format (matching C++ Dictionary::load).
    ///
    /// Format:
    /// - size: i32
    /// - nwords: i32
    /// - nlabels: i32
    /// - ntokens: i64
    /// - pruneidx_size: i64
    /// - For each entry: null-terminated word string + count(i64) + type(i8)
    /// - If pruneidx_size > 0: that many (i32, i32) pairs
    ///
    /// After loading, initializes discard table, n-grams, and rebuilds word2int.
    pub fn load_from_reader<R: Read>(reader: &mut R, args: Arc<Args>) -> Result<Self> {
        let size = utils::read_i32(reader)?;
        let nwords = utils::read_i32(reader)?;
        let nlabels = utils::read_i32(reader)?;
        let ntokens = utils::read_i64(reader)?;
        let pruneidx_size = utils::read_i64(reader)?;

        // Validate dimensions
        if size < 0 || nwords < 0 || nlabels < 0 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid dictionary dimensions: size={}, nwords={}, nlabels={}",
                size, nwords, nlabels
            )));
        }

        // Read vocabulary entries
        let mut words = Vec::with_capacity(size as usize);
        for _ in 0..size {
            // Read null-terminated string
            let mut word_bytes: Vec<u8> = Vec::new();
            let mut buf = [0u8; 1];
            loop {
                reader.read_exact(&mut buf)?;
                if buf[0] == 0 {
                    break;
                }
                word_bytes.push(buf[0]);
            }
            let word = String::from_utf8(word_bytes).map_err(|_| {
                FastTextError::InvalidModel("Invalid UTF-8 in dictionary word".to_string())
            })?;
            let count = utils::read_i64(reader)?;
            let mut type_buf = [0u8; 1];
            reader.read_exact(&mut type_buf)?;
            let entry_type = match type_buf[0] as i8 {
                0 => EntryType::Word,
                1 => EntryType::Label,
                v => {
                    return Err(FastTextError::InvalidModel(format!(
                        "Invalid entry type: {}",
                        v
                    )))
                }
            };
            words.push(Entry {
                word,
                count,
                entry_type,
                subwords: Vec::new(),
            });
        }

        // Read pruneidx pairs
        let mut pruneidx = HashMap::new();
        if pruneidx_size > 0 {
            for _ in 0..pruneidx_size {
                let first = utils::read_i32(reader)?;
                let second = utils::read_i32(reader)?;
                pruneidx.insert(first, second);
            }
        }

        // Build the dictionary with a tiny placeholder (will be resized by rebuild)
        let mut dict = Dictionary::new_with_capacity(args, 1);
        dict.words = words;
        dict.size = size;
        dict.nwords = nwords;
        dict.nlabels = nlabels;
        dict.ntokens = ntokens;
        dict.pruneidx_size = pruneidx_size;
        dict.pruneidx = pruneidx;

        // Initialize discard table and n-grams (matching C++ Dictionary::load order)
        dict.init_table_discard();
        dict.init_ngrams();

        // Rebuild word2int hash table with size = ceil(size / 0.7)
        dict.rebuild_word2int_after_load();

        Ok(dict)
    }

    /// Check whether the dictionary is pruned (pruneidx_size >= 0).
    pub fn is_pruned(&self) -> bool {
        self.pruneidx_size >= 0
    }

    /// Prune the dictionary to keep only the embeddings specified by `idx`.
    ///
    /// Matches C++ `Dictionary::prune`. The `idx` slice contains mixed word-row
    /// indices and bucket (ngram) row indices.  Word indices are `< nwords`;
    /// ngram indices are `>= nwords`.
    ///
    /// After pruning:
    /// - Only the words whose indices appear in `idx` are kept (labels are always kept).
    /// - Bucket n-gram rows are remapped via `pruneidx` so downstream code can
    ///   still look up their new positions in the pruned input matrix.
    /// - `nwords`, `size`, and `word2int` are updated accordingly.
    /// - `init_ngrams()` is called to recompute subwords with the new vocabulary.
    pub fn prune(&mut self, idx: &[i32]) {
        let mut words_idx: Vec<i32> = Vec::new();
        let mut ngrams_idx: Vec<i32> = Vec::new();

        for &i in idx {
            if i < self.nwords {
                words_idx.push(i);
            } else {
                ngrams_idx.push(i);
            }
        }
        words_idx.sort_unstable();

        // Build pruneidx: maps original bucket-relative ngram index to new position.
        // Ngram IDs in the input matrix are stored as nwords + relative_idx.
        // After pruning, bucket rows start after the pruned word rows.
        // pruneidx maps (original_ngram_id - nwords) → new_position_in_ngram_block
        self.pruneidx.clear();
        for (j, &ngram) in ngrams_idx.iter().enumerate() {
            self.pruneidx.insert(ngram - self.nwords, j as i32);
        }
        self.pruneidx_size = self.pruneidx.len() as i64;

        // Compact the words array: keep only selected words + all labels.
        // The C++ algorithm iterates words_ in order; word i is kept if its
        // original index appears in words_idx (sorted), or if it is a label.
        let mut new_words: Vec<Entry> = Vec::with_capacity(words_idx.len() + self.nlabels as usize);
        let mut word_ptr = 0usize; // pointer into words_idx (sorted)
        for i in 0..self.words.len() {
            let entry_type = self.words[i].entry_type;
            if entry_type == EntryType::Label {
                // Labels are always kept.
                new_words.push(self.words[i].clone());
            } else if word_ptr < words_idx.len() && words_idx[word_ptr] == i as i32 {
                // This word index was selected.
                new_words.push(self.words[i].clone());
                word_ptr += 1;
            }
        }

        // Reorder so that words come first, then labels (matching C++).
        let new_nwords = words_idx.len() as i32;
        let new_nlabels = self.nlabels;
        // Separate words and labels from new_words (labels were pushed after words above).
        let mut words_part: Vec<Entry> = Vec::new();
        let mut labels_part: Vec<Entry> = Vec::new();
        for e in new_words {
            if e.entry_type == EntryType::Word {
                words_part.push(e);
            } else {
                labels_part.push(e);
            }
        }
        // Words must be in sorted index order (already are since words_idx is sorted and we
        // iterated words_ in order).
        self.words = words_part;
        self.words.extend(labels_part);

        self.nwords = new_nwords;
        self.size = self.nwords + new_nlabels;

        // Rebuild word2int hash table.
        let word2int_size = ((self.size as f64 / 0.7).ceil() as usize).max(1);
        self.word2int = vec![-1i32; word2int_size];
        for i in 0..self.size as usize {
            let w = self.words[i].word.clone();
            let h = hash(w.as_bytes());
            let slot = self.find_slot_with_hash(&w, h);
            self.word2int[slot] = i as i32;
        }

        // Recompute subwords for the pruned vocabulary.
        self.init_ngrams();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Args;
    use std::sync::Arc;

    /// Create a default Args reference.
    fn make_args() -> Arc<Args> {
        Arc::new(Args::default())
    }

    /// Create a small Dictionary for tests (capacity 1024 to avoid 120MB allocation).
    fn make_dict() -> Dictionary {
        Dictionary::new_with_capacity(make_args(), 1024)
    }

    // =========================================================================
    // VAL-DICT-001: Tokenization rules
    // =========================================================================

    #[test]
    fn test_tokenize_whitespace() {
        // Space
        assert_eq!(Dictionary::tokenize("hello world"), vec!["hello", "world"]);
        // Tab
        assert_eq!(Dictionary::tokenize("hello\tworld"), vec!["hello", "world"]);
        // Vertical tab (\v = 0x0b)
        assert_eq!(
            Dictionary::tokenize("hello\x0bworld"),
            vec!["hello", "world"]
        );
        // Form feed (\f = 0x0c)
        assert_eq!(
            Dictionary::tokenize("hello\x0cworld"),
            vec!["hello", "world"]
        );
        // Null byte
        assert_eq!(Dictionary::tokenize("hello\0world"), vec!["hello", "world"]);
        // Carriage return (\r)
        assert_eq!(Dictionary::tokenize("hello\rworld"), vec!["hello", "world"]);
        // Multiple consecutive whitespace (collapsed)
        assert_eq!(
            Dictionary::tokenize("hello   world"),
            vec!["hello", "world"]
        );
        // Leading/trailing whitespace
        assert_eq!(
            Dictionary::tokenize("  hello world  "),
            vec!["hello", "world"]
        );
        // Mixed whitespace types
        assert_eq!(
            Dictionary::tokenize("hello \t world"),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn test_tokenize_eos() {
        // Newline produces EOS token
        assert_eq!(
            Dictionary::tokenize("hello\nworld"),
            vec!["hello", EOS, "world"]
        );
        // Newline at start
        assert_eq!(Dictionary::tokenize("\nhello"), vec![EOS, "hello"]);
        // Newline at end
        assert_eq!(Dictionary::tokenize("hello\n"), vec!["hello", EOS]);
        // Multiple consecutive newlines
        assert_eq!(
            Dictionary::tokenize("hello\n\nworld"),
            vec!["hello", EOS, EOS, "world"]
        );
        // \r\n: \r terminates word (non-newline whitespace), \n produces EOS.
        assert_eq!(
            Dictionary::tokenize("hello\r\nworld"),
            vec!["hello", EOS, "world"]
        );
        // a \n b pattern
        assert_eq!(Dictionary::tokenize("a\nb"), vec!["a", EOS, "b"]);
        // Multiple words, then newline
        assert_eq!(
            Dictionary::tokenize("hello world\nfoo bar"),
            vec!["hello", "world", EOS, "foo", "bar"]
        );
    }

    #[test]
    fn test_tokenize_max_tokens() {
        // 1025 space-separated words on one line → exactly 1024 returned
        let words: Vec<String> = (0..1025).map(|i| format!("word{}", i)).collect();
        let text = words.join(" ");
        let result = Dictionary::tokenize(&text);
        assert_eq!(
            result.len(),
            MAX_LINE_SIZE,
            "Expected {} tokens, got {}",
            MAX_LINE_SIZE,
            result.len()
        );

        // The first 1024 words should be present
        for (i, token) in result.iter().enumerate() {
            assert_eq!(token, &format!("word{}", i));
        }

        // After EOS, a new line gets a fresh count
        // "word0...word1024\nword0...word1024" → 1024 + EOS + 1024 = 2049
        let text2 = format!("{}\n{}", text, text);
        let result2 = Dictionary::tokenize(&text2);
        assert_eq!(result2.len(), 2 * MAX_LINE_SIZE + 1); // 1024 + EOS + 1024

        // The EOS should be at position 1024
        assert_eq!(result2[MAX_LINE_SIZE], EOS);
    }

    #[test]
    fn test_tokenize_utf8() {
        // UTF-8 multi-byte sequences preserved intact
        let result = Dictionary::tokenize("hello 日本語 world");
        assert_eq!(result, vec!["hello", "日本語", "world"]);

        // Accented characters
        let result = Dictionary::tokenize("café naïve");
        assert_eq!(result, vec!["café", "naïve"]);

        // Mixed ASCII and UTF-8 with tab separator
        let result = Dictionary::tokenize("hello\t日本語\ncafé");
        assert_eq!(result, vec!["hello", "日本語", EOS, "café"]);

        // Emoji
        let result = Dictionary::tokenize("hello 🎉 world");
        assert_eq!(result, vec!["hello", "🎉", "world"]);

        // Chinese characters
        let result = Dictionary::tokenize("你好 世界");
        assert_eq!(result, vec!["你好", "世界"]);
    }

    // =========================================================================
    // VAL-DICT-002: Label detection with configurable prefix
    // =========================================================================

    #[test]
    fn test_label_detection_default() {
        let dict = make_dict();

        // Default prefix __label__
        assert_eq!(
            dict.get_type_from_str("__label__cat"),
            EntryType::Label,
            "__label__cat should be a label"
        );
        assert_eq!(
            dict.get_type_from_str("__label__"),
            EntryType::Label,
            "__label__ alone should be a label"
        );
        assert_eq!(
            dict.get_type_from_str("__label__very_long_label_name"),
            EntryType::Label,
            "longer label should be detected"
        );

        // Non-labels
        assert_eq!(
            dict.get_type_from_str("hello"),
            EntryType::Word,
            "regular word should be Word"
        );
        assert_eq!(
            dict.get_type_from_str("label__"),
            EntryType::Word,
            "label__ without __ prefix should be Word"
        );
        assert_eq!(
            dict.get_type_from_str("_label_cat"),
            EntryType::Word,
            "single underscore prefix should be Word"
        );
        assert_eq!(
            dict.get_type_from_str(""),
            EntryType::Word,
            "empty string should be Word"
        );
        assert_eq!(
            dict.get_type_from_str(EOS),
            EntryType::Word,
            "EOS token should be Word"
        );
    }

    #[test]
    fn test_label_detection_custom_prefix() {
        let mut args = Args::default();
        args.set_label("#".to_string());
        let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        assert_eq!(
            dict.get_type_from_str("#cat"),
            EntryType::Label,
            "#cat should be a label with # prefix"
        );
        assert_eq!(
            dict.get_type_from_str("#"),
            EntryType::Label,
            "# alone should be a label"
        );
        assert_eq!(
            dict.get_type_from_str("cat"),
            EntryType::Word,
            "cat should be a word"
        );
        // Old default prefix should not match
        assert_eq!(
            dict.get_type_from_str("__label__cat"),
            EntryType::Word,
            "__label__cat should be Word with # prefix"
        );
    }

    // =========================================================================
    // VAL-DICT-003: Vocabulary management
    // =========================================================================

    #[test]
    fn test_vocab_lookup() {
        let mut dict = make_dict();
        dict.add("hello");
        dict.add("world");
        dict.add("hello");

        let hello_id = dict.get_id("hello");
        let world_id = dict.get_id("world");
        let oov_id = dict.get_id("foo");

        assert!(hello_id >= 0, "hello should be in vocab");
        assert!(world_id >= 0, "world should be in vocab");
        assert_ne!(
            hello_id, world_id,
            "hello and world should have different IDs"
        );
        assert_eq!(oov_id, -1, "foo should be OOV");

        // Lookup by ID
        assert_eq!(dict.get_word(hello_id), "hello");
        assert_eq!(dict.get_word(world_id), "world");
    }

    #[test]
    fn test_vocab_counts() {
        let mut dict = make_dict();
        dict.add("hello");
        dict.add("world");
        dict.add("hello");
        dict.add("hello");

        // ntokens counts every add() call
        assert_eq!(dict.ntokens(), 4);
        // size is number of distinct entries
        assert_eq!(dict.size(), 2);
        // nwords/nlabels not updated until threshold()
        dict.threshold(1, 1);
        assert_eq!(dict.nwords(), 2);
        assert_eq!(dict.nlabels(), 0);
        assert_eq!(dict.size(), 2);
    }

    #[test]
    fn test_vocab_label_counts() {
        let mut dict = make_dict();
        dict.add("word1");
        dict.add("__label__cat");
        dict.add("word2");
        dict.add("__label__dog");
        dict.add("__label__cat");

        assert_eq!(dict.ntokens(), 5);

        dict.threshold(1, 1);
        assert_eq!(dict.nwords(), 2);
        assert_eq!(dict.nlabels(), 2);
        assert_eq!(dict.size(), 4);

        // Label IDs should be accessible.
        // __label__cat (count 2) sorts before __label__dog (count 1) in descending order.
        assert_eq!(dict.get_label(0).unwrap(), "__label__cat"); // count 2, sorted first
        assert_eq!(dict.get_label(1).unwrap(), "__label__dog"); // count 1, sorted second
    }

    #[test]
    fn test_vocab_sorted_order() {
        let mut dict = make_dict();
        // Add words with different frequencies
        for _ in 0..5 {
            dict.add("rare");
        } // count 5
        for _ in 0..10 {
            dict.add("common");
        } // count 10
        for _ in 0..3 {
            dict.add("__label__cat");
        } // label count 3
        for _ in 0..7 {
            dict.add("__label__dog");
        } // label count 7

        dict.threshold(1, 1);

        // Words should come before labels
        assert_eq!(dict.nwords(), 2, "Should have 2 words");
        assert_eq!(dict.nlabels(), 2, "Should have 2 labels");

        // Within words: descending count
        assert_eq!(dict.words()[0].word, "common");
        assert_eq!(dict.words()[0].count, 10);
        assert_eq!(dict.words()[1].word, "rare");
        assert_eq!(dict.words()[1].count, 5);

        // Within labels: descending count
        assert_eq!(dict.words()[2].word, "__label__dog");
        assert_eq!(dict.words()[2].count, 7);
        assert_eq!(dict.words()[3].word, "__label__cat");
        assert_eq!(dict.words()[3].count, 3);

        // IDs should be accessible via get_id
        assert_eq!(dict.get_id("common"), 0);
        assert_eq!(dict.get_id("rare"), 1);
        assert_eq!(dict.get_id("__label__dog"), 2);
        assert_eq!(dict.get_id("__label__cat"), 3);
    }

    #[test]
    fn test_vocab_threshold_filtering() {
        let mut dict = make_dict();
        dict.add("rare"); // count 1
        dict.add("common"); // count 2
        dict.add("common");
        dict.add("__label__a"); // label count 1
        dict.add("__label__b"); // label count 2
        dict.add("__label__b");

        // Threshold: keep words with count >= 2, labels with count >= 1
        dict.threshold(2, 1);

        assert_eq!(dict.nwords(), 1, "Only 'common' survives");
        assert_eq!(dict.nlabels(), 2, "Both labels survive");

        assert!(dict.get_id("common") >= 0, "common should be in vocab");
        assert_eq!(dict.get_id("rare"), -1, "rare should be filtered out");
        assert!(dict.get_id("__label__a") >= 0, "label a should be in vocab");
        assert!(dict.get_id("__label__b") >= 0, "label b should be in vocab");
    }

    #[test]
    fn test_vocab_hash_collision_resolution() {
        // Test that the hash table correctly handles collisions via linear probing.
        // We'll add enough words to ensure some slots are probed.
        let mut dict = make_dict();

        // Add 100 distinct words
        for i in 0..100 {
            dict.add(&format!("word_{}", i));
        }

        assert_eq!(dict.size(), 100);

        // All words should be findable
        for i in 0..100 {
            let id = dict.get_id(&format!("word_{}", i));
            assert!(id >= 0, "word_{} should be in vocab", i);
        }

        // OOV should return -1
        assert_eq!(dict.get_id("not_in_vocab"), -1);
    }

    #[test]
    fn test_vocab_eos_is_word_type() {
        let dict = make_dict();
        // EOS is not a label
        assert_eq!(dict.get_type_from_str(EOS), EntryType::Word);
    }

    #[test]
    fn test_get_label_out_of_range() {
        let mut dict = make_dict();
        dict.add("__label__cat");
        dict.threshold(1, 1);

        // Valid label ID
        assert!(dict.get_label(0).is_ok());
        // Invalid label IDs
        assert!(dict.get_label(-1).is_err());
        assert!(dict.get_label(1).is_err());
    }

    // =========================================================================
    // Additional edge case tests
    // =========================================================================

    #[test]
    fn test_tokenize_empty_string() {
        let result = Dictionary::tokenize("");
        assert!(result.is_empty(), "Empty string should yield no tokens");
    }

    #[test]
    fn test_tokenize_only_whitespace() {
        let result = Dictionary::tokenize("   \t  ");
        assert!(result.is_empty(), "Whitespace-only should yield no tokens");
    }

    #[test]
    fn test_tokenize_only_newlines() {
        let result = Dictionary::tokenize("\n\n\n");
        assert_eq!(result, vec![EOS, EOS, EOS]);
    }

    #[test]
    fn test_add_increments_ntokens() {
        let mut dict = make_dict();
        assert_eq!(dict.ntokens(), 0);
        dict.add("a");
        assert_eq!(dict.ntokens(), 1);
        dict.add("b");
        assert_eq!(dict.ntokens(), 2);
        dict.add("a"); // Duplicate
        assert_eq!(dict.ntokens(), 3);
    }

    #[test]
    fn test_word_count_accumulation() {
        let mut dict = make_dict();
        for _ in 0..5 {
            dict.add("hello");
        }
        dict.threshold(1, 1);

        let id = dict.get_id("hello");
        assert!(id >= 0);
        assert_eq!(dict.words()[id as usize].count, 5);
    }

    #[test]
    fn test_read_word_from_reader_basic() {
        let text = "hello world\nfoo";
        let mut reader = text.as_bytes();
        let mut pending = false;
        let mut word = String::new();

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, "hello");

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, "world");

        // Newline was pending → EOS
        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, EOS);

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, "foo");

        // EOF
        assert!(!Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
    }

    #[test]
    fn test_read_word_from_reader_utf8() {
        // Multi-byte UTF-8 sequences must be preserved intact.
        // '日' = 0xE6 0x97 0xA5, 'é' in café = 0xC3 0xA9
        let text = "日本語 café\nhello";
        let mut reader = text.as_bytes();
        let mut pending = false;
        let mut word = String::new();

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(
            word, "日本語",
            "Multi-byte UTF-8 token '日本語' should be preserved intact"
        );

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(
            word, "café",
            "UTF-8 token 'café' with accented character should be preserved intact"
        );

        // Newline was pending → EOS
        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, EOS);

        assert!(Dictionary::read_word_from_reader(
            &mut reader,
            &mut pending,
            &mut word
        ));
        assert_eq!(word, "hello");
    }

    #[test]
    fn test_read_from_file_utf8_tokens() {
        // Verify that read_from_file (which uses read_word_from_reader) correctly
        // preserves multi-byte UTF-8 tokens in the vocabulary.
        let mut args = Args::default();
        args.set_min_count(1);
        let args = Arc::new(args);
        let mut dict = Dictionary::new_with_capacity(args, 1024);

        // Use an in-memory buffer simulating a training text file with UTF-8 tokens.
        let content = "日本語 café hello\n日本語 world\ncafé test\n";
        let mut reader = content.as_bytes();
        dict.read_from_file(&mut reader).unwrap();

        // All tokens should appear correctly in the vocabulary.
        let id_jp = dict.get_id("日本語");
        let id_cafe = dict.get_id("café");
        let id_hello = dict.get_id("hello");
        let id_world = dict.get_id("world");
        let id_test = dict.get_id("test");

        assert!(id_jp >= 0, "'日本語' should be in vocabulary (got -1)");
        assert!(id_cafe >= 0, "'café' should be in vocabulary (got -1)");
        assert!(id_hello >= 0, "'hello' should be in vocabulary (got -1)");
        assert!(id_world >= 0, "'world' should be in vocabulary (got -1)");
        assert!(id_test >= 0, "'test' should be in vocabulary (got -1)");

        // Verify the word strings are stored correctly (not corrupted).
        assert_eq!(
            dict.get_word(id_jp),
            "日本語",
            "Stored word for id_jp should be '日本語'"
        );
        assert_eq!(
            dict.get_word(id_cafe),
            "café",
            "Stored word for id_café should be 'café'"
        );

        // Verify frequencies (日本語 appears twice, café appears twice).
        assert_eq!(
            dict.words()[id_jp as usize].count,
            2,
            "'日本語' should have count 2"
        );
        assert_eq!(
            dict.words()[id_cafe as usize].count,
            2,
            "'café' should have count 2"
        );
    }

    #[test]
    fn test_tokenize_eos_entry_type() {
        // After building vocab from text with EOS, EOS should be a Word type
        let mut dict = make_dict();
        dict.add(EOS);
        dict.add("hello");
        dict.threshold(1, 1);

        let eos_id = dict.get_id(EOS);
        assert!(eos_id >= 0);
        assert_eq!(dict.get_type_by_id(eos_id), EntryType::Word);
    }

    // =========================================================================
    // VAL-DICT-004: Subword computation
    // =========================================================================

    /// Build an args with specific subword settings for testing.
    fn make_subword_args(minn: i32, maxn: i32, bucket: i32) -> Arc<Args> {
        let mut args = Args::default();
        args.set_minn(minn);
        args.set_maxn(maxn);
        args.set_bucket(bucket);
        Arc::new(args)
    }

    #[test]
    fn test_subword_computation_bow_eow_wrapping() {
        // The subword algorithm wraps with BOW '<' and EOW '>'.
        // With minn=3, maxn=4, bucket=100000, word="he":
        // Wrapped: "<he>" (4 bytes, all ASCII)
        // n-grams (len counted in Unicode chars):
        //   i=0 ('<'): n=1 → "<" skip (i==0, n==1); n=2 → "<h" < minn=3 skip; n=3 → "<he" include; n=4 → "<he>" include
        //   i=1 ('h'): n=1 → "h" < minn skip; n=2 → "he" < minn skip; n=3 → "he>" include; n=4 → j=4=size, loop ends
        //   i=2 ('e'): n=1 → "e" skip; n=2 → "e>" j=4=size, n=2<3 skip; n=3 → j=4 loop ends
        //   i=3 ('>'): n=1 → ">" j=4=size → n==1 && j==size → skip; loop ends
        // Total n-grams: "<he", "<he>", "he>" = 3
        let args = make_subword_args(3, 4, 100000);
        let dict = Dictionary::new_with_capacity(args, 1024);

        let mut ngrams = Vec::new();
        dict.compute_subwords("<he>", &mut ngrams);
        assert_eq!(
            ngrams.len(),
            3,
            "Should have 3 n-grams for '<he>' with minn=3 maxn=4"
        );
    }

    #[test]
    fn test_subword_bucket_index() {
        // All subword IDs should be in [nwords, nwords + bucket).
        let args = make_subword_args(3, 6, 200000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.add("world");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let nwords = dict.nwords();
        let bucket = 200000;

        for wid in 0..nwords {
            let subwords = dict.get_subwords(wid);
            // First element is the word ID itself.
            assert_eq!(subwords[0], wid, "First subword should be word ID");
            // Remaining elements are n-gram hashes mapped to buckets.
            for &sid in &subwords[1..] {
                assert!(
                    sid >= nwords,
                    "Subword ID {} should be >= nwords {}",
                    sid,
                    nwords
                );
                assert!(
                    sid < nwords + bucket,
                    "Subword ID {} should be < nwords+bucket {}",
                    sid,
                    nwords + bucket
                );
            }
        }
    }

    #[test]
    fn test_subword_computation_known_values() {
        // Verify that compute_subwords produces deterministic results.
        // With default minn=3, maxn=6, bucket=2000000, word "hello":
        // <hello> has these n-grams (n=3 to 6):
        // From i=0: <he, <hel, <hell, <hello
        // From i=1: hel, hell, hello, hello>
        // From i=2: ell, ello, ello>
        // From i=3: llo, llo>
        // From i=4: lo>
        // Total: 15 n-grams
        let args = make_subword_args(3, 6, 2_000_000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let wid = dict.get_id("hello");
        assert!(wid >= 0);
        let subwords = dict.get_subwords(wid);

        // 1 (word ID) + 14 (n-grams) = 15 total
        // See detailed count below in the comment block.

        // Verify the n-grams match manual computation.
        let ngram_strings = [
            "<he",
            "<hel",
            "<hell",
            "<hello",
            "hel",
            "hell",
            "hello",
            "hello>",
            "ell",
            "ello",
            "ello>",
            "llo",
            "llo>",
            "lo>",
            // Note: "lo" alone would be from i=5 ('o') but 'o' + '>' has only 2 chars → skip
            // Actually wait, from i=4 we also get lo> as shown above
            // Let me recount: from i=4 (second 'l'):
            //   n=1: "l" → skip, n=2: "lo" → skip, n=3: "lo>" → include
            // But wait from i=3 (first 'l'):
            //   n=1: "l" skip, n=2: "ll" skip, n=3: "llo" include, n=4: "llo>" include
            // From i=4 ('l' second):
            //   n=1: "l" skip, n=2: "lo" skip, n=3: "lo>" include
            // Total: 4+4+3+2+1 = 14... hmm let me recount
            // i=0 '<': n=3,4,5,6 → 4 ngrams
            // i=1 'h': n=3,4,5,6 → 4 ngrams
            // i=2 'e': n=3,4,5 → 3 ngrams (n=6 would need j+5 chars but only 4 remain)
            // i=3 'l': n=3,4 → 2 ngrams (n=5 would need j+4 but only 3 remain)
            // i=4 'l': n=3 → 1 ngram
            // i=5 'o': n=3? j starts at 5, push 'o', j=6, push '>', j=7=size. n=2 → only 2 chars → skip
            // i=6 '>': n=1, j=7=size → skip
            // Total: 4+4+3+2+1 = 14 n-grams
            "unused_for_count_only",
        ];

        // Use 14 based on the correct count above.
        // The test above uses 16 (1 word ID + 15) - let me recount again carefully.
        // <hello> has 7 bytes: '<'=0, 'h'=1, 'e'=2, 'l'=3, 'l'=4, 'o'=5, '>'=6

        // i=0 ('<'), j=0: outer pos 0
        //   n=1: push '<', j=1. Skip (n==1 && i==0).
        //   n=2: push 'h', j=2. ngram="<h". n=2 < minn=3: skip.
        //   n=3: push 'e', j=3. ngram="<he". n=3>=3, not n==1: INCLUDE.
        //   n=4: push 'l', j=4. ngram="<hel". INCLUDE.
        //   n=5: push 'l', j=5. ngram="<hell". INCLUDE.
        //   n=6: push 'o', j=6. ngram="<hello". INCLUDE. (j=6 < 7)
        //   n=7: j < n_bytes (6 < 7), but n > maxn=6: loop ends.
        // Wait: the loop is `while j < n_bytes && n <= maxn`. After n=6, we check j < 7 (6 < 7, true) AND n <= 6 (7 <= 6, false). So loop ends after n=6.
        // But wait, n was 6 at the last iteration. The next check would be n=7 > maxn=6, so we stop.
        // So from i=0: n=3,4,5,6 → ngrams: "<he", "<hel", "<hell", "<hello" = 4 ngrams.

        // i=1 ('h'), j=1:
        //   n=1: push 'h', j=2. "h". n=1 < minn: skip.
        //   n=2: push 'e', j=3. "he". n=2 < minn: skip.
        //   n=3: push 'l', j=4. "hel". n=3, include.
        //   n=4: push 'l', j=5. "hell". include.
        //   n=5: push 'o', j=6. "hello". include. (j=6 < 7)
        //   n=6: push '>', j=7. "hello>". j=7=n_bytes. n=6. NOT (n==1 && j==n_bytes). include!
        //   n=7 > maxn: loop ends.
        // From i=1: "hel", "hell", "hello", "hello>" = 4 ngrams.

        // i=2 ('e'), j=2:
        //   n=1: "e". skip.
        //   n=2: "el". skip.
        //   n=3: "ell". include.
        //   n=4: "ello". include.
        //   n=5: "ello>". j=7. n=5, NOT n==1: include.
        //   n=6: j=7 >= n_bytes: loop ends.
        // From i=2: "ell", "ello", "ello>" = 3 ngrams.

        // i=3 ('l'), j=3:
        //   n=1: "l". skip.
        //   n=2: "ll". skip.
        //   n=3: "llo". include.
        //   n=4: "llo>". j=7. n=4, NOT n==1: include.
        //   n=5: j=7 >= n_bytes: loop ends.
        // From i=3: "llo", "llo>" = 2 ngrams.

        // i=4 ('l' second), j=4:
        //   n=1: "l". skip.
        //   n=2: "lo". skip.
        //   n=3: "lo>". j=7. n=3, NOT n==1: include.
        //   n=4: j=7 >= size: loop ends.
        // From i=4: "lo>" = 1 ngram.

        // i=5 ('o'), j=5:
        //   n=1: "o". skip.
        //   n=2: "o>". j=7. n=2 < minn=3: skip.
        //   n=3: j=7 >= size: loop ends.
        // From i=5: 0 ngrams.

        // i=6 ('>'), j=6:
        //   n=1: ">". j=7. n==1 && j==n_bytes → skip.
        //   n=2: j=7 >= size: loop ends.
        // From i=6: 0 ngrams.

        // Total: 4+4+3+2+1 = 14 n-grams.
        // With word ID prepended: 15 total.
        let _ = ngram_strings; // suppress unused warning

        // 1 word ID + 14 n-grams = 15 total.
        assert_eq!(
            subwords.len(),
            15,
            "Expected 15 subwords for 'hello' (1 word ID + 14 n-grams): got {:?}",
            subwords
        );
    }

    #[test]
    fn test_subword_computation_utf8_aware() {
        // Test with a word containing multi-byte UTF-8 characters.
        // "café" = c(0x63), a(0x61), f(0x66), é(0xC3 0xA9)
        // With BOW/EOW: "<café>" = '<' + 'c' + 'a' + 'f' + 'é'(2bytes) + '>' = 8 bytes, 6 chars
        //
        // With minn=1, maxn=3:
        // The outer loop skips continuation bytes (0x80-0xBF)
        // é has leading byte 0xC3 (not continuation) and continuation byte 0xA9
        //
        // n-grams of Unicode length 1, 2, 3:
        // At byte i=0 ('<'): len=1 "< " → skip (i==0, n==1)
        //                    len=2 "<c" → include if minn=1 allows... wait minn=1
        //                    Actually with minn=1:
        //                    n=1: "<" → skip (i==0, n==1)
        //                    n=2: "<c" → include (n>=1 and NOT (n==1 and boundary))
        //                    n=3: "<ca" → include
        // With minn=1, we get many more n-grams.
        // Let's use minn=2, maxn=3 to keep it manageable.
        // "café" has 4 Unicode chars: c,a,f,é. With markers: "<café>" = 6 Unicode chars.
        //
        // The key test here is that 'é' (2 bytes) counts as 1 Unicode char, not 2.
        // n-grams: starting from outer positions (byte offsets of Unicode char starts):
        // pos 0: '<' (1 byte)
        // pos 1: 'c' (1 byte)
        // pos 2: 'a' (1 byte)
        // pos 3: 'f' (1 byte)
        // pos 4: 'é' (2 bytes: 0xC3, 0xA9)
        // pos 6: '>' (1 byte)

        let args = make_subword_args(2, 3, 100000);
        let dict = Dictionary::new_with_capacity(args, 1024);

        // Compute for "<café>"
        let word_with_markers = format!("{}{}{}", BOW, "café", EOW);
        let mut ngrams = Vec::new();
        dict.compute_subwords(&word_with_markers, &mut ngrams);

        // Each ngram ID should be in valid range [0, bucket) (nwords=0, so IDs are in [0, 100000)).
        for &id in &ngrams {
            assert!(
                id >= 0 && id < 100000,
                "N-gram ID {} out of range [0, 100000)",
                id
            );
        }

        // Verify count: with minn=2, maxn=3, 6 Unicode chars in "<café>":
        // From each of 6 positions, n-grams of length 2 and 3 (where possible):
        // pos 0 '<': n=1 → skip (i==0, n==1), n=2 → "<c" include, n=3 → "<ca" include
        // pos 1 'c': n=1 skip (n<minn=2 when minn=2), n=2 → "ca" include, n=3 → "caf" include
        // pos 2 'a': n=2 → "af" include, n=3 → "afé" include
        // pos 3 'f': n=2 → "fé" include, n=3 → "fé>" include
        // pos 4 'é' (0xC3 0xA9): n=2 → "é>" include, n=3 → j past end: end
        //   Actually wait: after consuming 'é' (2 bytes), j moves to 6. Then '>' is at 6.
        //   n=1: push é (bytes 4,5), j=6. n=1. Check: n>=minn(2)? No. Skip.
        //   n=2: push '>' (byte 6), j=7. n=2>=2, j=7=n_bytes → n==1? No. Include. "é>" ✓
        //   n=3: j=7 >= n_bytes: ends.
        // pos 6 '>': n=1: push '>', j=7=n_bytes → skip (n==1 && j==n_bytes)
        // Total: 2 + 2 + 2 + 2 + 1 = 9 n-grams
        assert_eq!(
            ngrams.len(),
            9,
            "Expected 9 n-grams for '<café>' with minn=2 maxn=3"
        );

        // Verify the n-grams involving 'é' use raw bytes (UTF-8 bytes, not codepoints).
        // The hash of "fé>" should use the raw bytes of é (0xC3, 0xA9).
        let expected_hash_of_f_e_gt = (crate::utils::hash("fé>".as_bytes()) % 100000) as i32;
        assert!(
            ngrams.contains(&expected_hash_of_f_e_gt),
            "N-grams should contain hash('fé>') = {}",
            expected_hash_of_f_e_gt
        );
    }

    // =========================================================================
    // VAL-DICT-005: Subword edge cases
    // =========================================================================

    #[test]
    fn test_subword_eos_no_subwords() {
        // EOS should only have its own ID, no subword n-grams.
        let args = make_subword_args(3, 6, 2_000_000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add(EOS);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let eos_id = dict.get_id(EOS);
        assert!(eos_id >= 0, "EOS should be in vocab");

        let subwords = dict.get_subwords(eos_id);
        assert_eq!(
            subwords.len(),
            1,
            "EOS should have only 1 entry (its own ID)"
        );
        assert_eq!(subwords[0], eos_id, "EOS subwords[0] should be its own ID");
    }

    #[test]
    fn test_subword_zero_bucket() {
        // When bucket=0, subword computation is disabled.
        // init_ngrams should produce only [word_id] for every word.
        let args = make_subword_args(3, 6, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.add("world");
        dict.threshold(1, 1);
        dict.init_ngrams();

        for wid in 0..dict.nwords() {
            let subwords = dict.get_subwords(wid);
            assert_eq!(
                subwords.len(),
                1,
                "With bucket=0, word {} should have only 1 subword",
                dict.get_word(wid)
            );
            assert_eq!(subwords[0], wid, "Subwords[0] should be the word ID");
        }
    }

    #[test]
    fn test_subword_zero_maxn() {
        // When maxn=0, no n-grams should be computed.
        let args = make_subword_args(0, 0, 2_000_000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let wid = dict.get_id("hello");
        let subwords = dict.get_subwords(wid);
        assert_eq!(
            subwords.len(),
            1,
            "With maxn=0, only word ID should be in subwords"
        );
    }

    #[test]
    fn test_subword_compute_subwords_direct() {
        // Direct test of compute_subwords with zero bucket → empty output.
        let args = make_subword_args(3, 6, 0);
        let dict = Dictionary::new_with_capacity(args, 1024);
        let mut ngrams = Vec::new();
        dict.compute_subwords("<hello>", &mut ngrams);
        assert!(
            ngrams.is_empty(),
            "compute_subwords with bucket=0 should produce nothing"
        );
    }

    #[test]
    fn test_subword_compute_subwords_maxn_zero() {
        let args = make_subword_args(0, 0, 100);
        let dict = Dictionary::new_with_capacity(args, 1024);
        let mut ngrams = Vec::new();
        dict.compute_subwords("<hello>", &mut ngrams);
        assert!(
            ngrams.is_empty(),
            "compute_subwords with maxn=0 should produce nothing"
        );
    }

    // =========================================================================
    // VAL-DICT-006: Word n-gram hashing
    // =========================================================================

    #[test]
    fn test_word_ngram_hash_bigram() {
        // Verify the rolling hash formula for a bigram.
        // h = hash(word1) (as i64 as u64), then h = h * 116049371 + hash(word2)
        // Result pushed is nwords + (h % bucket).
        let mut args = Args::default();
        args.set_bucket(100000);
        args.set_word_ngrams(2);
        let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        let h1 = crate::utils::hash(b"hello") as i32;
        let h2 = crate::utils::hash(b"world") as i32;
        let hashes = vec![h1, h2];

        let mut line = Vec::new();
        dict.add_word_ngrams(&mut line, &hashes, 2);

        // Should add exactly one bigram (1 pair).
        assert_eq!(line.len(), 1, "Bigram should produce exactly one hash");

        // Manually verify the formula:
        let expected_h = (h1 as i64 as u64)
            .wrapping_mul(116049371u64)
            .wrapping_add(h2 as i64 as u64);
        let expected_id = (expected_h % 100000) as i32;
        // nwords = 0 in empty dict, so result is 0 + expected_id = expected_id.
        assert_eq!(
            line[0], expected_id,
            "Bigram hash mismatch: got {}, expected {}",
            line[0], expected_id
        );
    }

    #[test]
    fn test_word_ngram_hash_trigram() {
        // 3 words with wordNgrams=3: bigrams (1,2), (2,3) + trigram (1,2,3).
        let mut args = Args::default();
        args.set_bucket(1_000_000);
        args.set_word_ngrams(3);
        let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        let h1 = crate::utils::hash(b"the") as i32;
        let h2 = crate::utils::hash(b"quick") as i32;
        let h3 = crate::utils::hash(b"brown") as i32;
        let hashes = vec![h1, h2, h3];

        let mut line = Vec::new();
        dict.add_word_ngrams(&mut line, &hashes, 3);

        // i=0: j=1 → bigram(h1,h2), j=2 → trigram(h1,h2,h3)
        // i=1: j=2 → bigram(h2,h3)
        // i=2: no j available
        // Total: 3 hashes
        assert_eq!(
            line.len(),
            3,
            "3 words with wordNgrams=3 should produce 3 hashes"
        );

        // Verify bigram (h1, h2):
        let h12 = (h1 as i64 as u64)
            .wrapping_mul(116049371u64)
            .wrapping_add(h2 as i64 as u64);
        let id12 = (h12 % 1_000_000) as i32;
        assert_eq!(line[0], id12, "First hash should be bigram (h1,h2)");

        // Verify trigram (h1, h2, h3):
        let h123 = h12
            .wrapping_mul(116049371u64)
            .wrapping_add(h3 as i64 as u64);
        let id123 = (h123 % 1_000_000) as i32;
        assert_eq!(line[1], id123, "Second hash should be trigram (h1,h2,h3)");

        // Verify bigram (h2, h3):
        let h23 = (h2 as i64 as u64)
            .wrapping_mul(116049371u64)
            .wrapping_add(h3 as i64 as u64);
        let id23 = (h23 % 1_000_000) as i32;
        assert_eq!(line[2], id23, "Third hash should be bigram (h2,h3)");
    }

    #[test]
    fn test_word_ngram_no_ngrams_for_word_ngrams_1() {
        // With wordNgrams=1, no n-grams should be added.
        let mut args = Args::default();
        args.set_bucket(100000);
        args.set_word_ngrams(1);
        let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        let hashes = vec![
            crate::utils::hash(b"hello") as i32,
            crate::utils::hash(b"world") as i32,
        ];
        let mut line = Vec::new();
        dict.add_word_ngrams(&mut line, &hashes, 1);

        assert!(line.is_empty(), "wordNgrams=1 should produce no n-grams");
    }

    #[test]
    fn test_word_ngram_zero_bucket() {
        // With bucket=0, add_word_ngrams is a no-op.
        let mut args = Args::default();
        args.set_bucket(0);
        args.set_word_ngrams(2);
        let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        let hashes = vec![
            crate::utils::hash(b"hello") as i32,
            crate::utils::hash(b"world") as i32,
        ];
        let mut line = Vec::new();
        dict.add_word_ngrams(&mut line, &hashes, 2);

        assert!(
            line.is_empty(),
            "bucket=0 should disable word n-gram hashing"
        );
    }

    #[test]
    fn test_word_ngram_ids_in_range() {
        // All generated n-gram IDs should be in [nwords, nwords + bucket).
        let mut args = Args::default();
        args.set_bucket(500);
        args.set_word_ngrams(3);
        let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);
        dict.add("a");
        dict.add("b");
        dict.add("c");
        dict.threshold(1, 1);

        let hashes: Vec<i32> = ["a", "b", "c"]
            .iter()
            .map(|w| crate::utils::hash(w.as_bytes()) as i32)
            .collect();
        let mut line = Vec::new();
        dict.add_word_ngrams(&mut line, &hashes, 3);

        let nwords = dict.nwords();
        for &id in &line {
            assert!(
                id >= nwords,
                "N-gram ID {} should be >= nwords {}",
                id,
                nwords
            );
            assert!(
                id < nwords + 500,
                "N-gram ID {} should be < nwords+bucket {}",
                id,
                nwords + 500
            );
        }
    }

    // =========================================================================
    // VAL-DICT-007: Subsampling discard probability
    // =========================================================================

    #[test]
    fn test_discard_table_formula() {
        // Build a dictionary with known frequencies and verify pdiscard formula.
        // pdiscard[i] = sqrt(t / f) + t / f  where f = count / ntokens.
        // With t=0.0001:
        //   If count=100, ntokens=1000 → f=0.1 → pdiscard = sqrt(0.001) + 0.001 ≈ 0.03262
        let mut args = Args::default();
        args.set_t(0.0001);
        let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        // Add "word" 100 times and 900 "other"s → ntokens = 1000, word frequency = 0.1
        for _ in 0..100 {
            dict.add("word");
        }
        for _ in 0..900 {
            dict.add("other");
        }
        dict.threshold(1, 1);
        // After threshold, init_table_discard is called inside read_from_file,
        // but here we call it directly.
        dict.init_table_discard();

        let wid = dict.get_id("word");
        assert!(wid >= 0);

        let pdiscard = dict.get_discard(wid);
        // f = 100 / 1000 = 0.1
        // expected = sqrt(0.0001 / 0.1) + 0.0001 / 0.1 = sqrt(0.001) + 0.001
        let f = 100.0f32 / 1000.0f32;
        let t = 0.0001f32;
        let expected = (t / f).sqrt() + t / f;
        assert!(
            (pdiscard - expected).abs() < 1e-5,
            "pdiscard {} should be close to expected {}",
            pdiscard,
            expected
        );
    }

    #[test]
    fn test_discard_supervised_bypass() {
        // In supervised mode, discard() always returns false.
        let mut args = Args::default();
        args.set_t(1.0); // Very high threshold: would discard everything
        args.apply_supervised_defaults();
        let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        for _ in 0..3 {
            dict.add("word");
        }
        dict.threshold(1, 1);
        dict.init_table_discard();

        let wid = dict.get_id("word");
        assert!(wid >= 0);

        // Even with rand=1.0 (which would normally trigger discard), supervised returns false.
        assert!(
            !dict.discard(wid, 1.0),
            "Supervised mode should never discard"
        );
        assert!(
            !dict.discard(wid, 0.5),
            "Supervised mode should never discard"
        );
        assert!(
            !dict.discard(wid, 0.0),
            "Supervised mode should never discard"
        );
    }

    #[test]
    fn test_discard_unsupervised_formula() {
        // In unsupervised mode, discard(id, rand) returns true when rand > pdiscard[id].
        // Use a very low-frequency word with high t to force discard.
        let mut args = Args::default();
        args.set_t(0.9); // Very high t
        args.set_model(crate::args::ModelName::SG); // Unsupervised
        let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

        // Add "rare" once, "common" 1000 times → f_rare ≈ 0.001
        dict.add("rare");
        for _ in 0..1000 {
            dict.add("common");
        }
        dict.threshold(1, 1);
        dict.init_table_discard();

        let wid = dict.get_id("rare");
        assert!(wid >= 0);

        // pdiscard["rare"] = sqrt(0.9 / (1/1001)) + 0.9 / (1/1001) ≈ very large (> 1.0)
        // So rand > pdiscard is false for any rand in [0,1], meaning it won't be discarded.
        let pdiscard_rare = dict.get_discard(wid);
        assert!(
            pdiscard_rare > 1.0,
            "Very rare word with high t should have pdiscard > 1.0"
        );

        // With rand=0.5 < pdiscard, discard returns false (not discarded).
        assert!(
            !dict.discard(wid, 0.5),
            "Should not discard rare word with rand=0.5"
        );
        // With rand=0.99 < pdiscard_rare, discard returns false.
        assert!(
            !dict.discard(wid, 0.99),
            "Should not discard rare word with rand=0.99"
        );

        let _wid_common = dict.get_id("common");
        // "common" has high frequency → low pdiscard.
        // f = 1000/1001 ≈ 0.999
        // pdiscard = sqrt(0.9/0.999) + 0.9/0.999 ≈ sqrt(0.901) + 0.901 ≈ 0.949 + 0.901 ≈ 1.85
        // Still > 1.0... Let's use a very high freq to get a small pdiscard.
        // With t=0.0001, f=0.999: pdiscard = sqrt(0.0001) + 0.0001 ≈ 0.01001
        // rand=0.5 > 0.01001 → discard returns true.
        let mut args2 = Args::default();
        args2.set_t(0.0001);
        args2.set_model(crate::args::ModelName::SG);
        let mut dict2 = Dictionary::new_with_capacity(Arc::new(args2), 1024);
        for _ in 0..9999 {
            dict2.add("frequent");
        }
        dict2.add("rare2");
        dict2.threshold(1, 1);
        dict2.init_table_discard();

        let wid_frequent = dict2.get_id("frequent");
        let pdiscard_freq = dict2.get_discard(wid_frequent);
        // f = 9999/10000 = 0.9999
        // pdiscard = sqrt(0.0001/0.9999) + 0.0001/0.9999 ≈ 0.01 + 0.0001 ≈ 0.0101
        assert!(
            pdiscard_freq < 1.0,
            "Frequent word should have pdiscard < 1.0, got {}",
            pdiscard_freq
        );
        // rand=0.5 > pdiscard_freq → discard returns true.
        assert!(
            dict2.discard(wid_frequent, 0.5),
            "Should discard frequent word with rand=0.5 > pdiscard"
        );
        // rand=0.001 < pdiscard_freq → discard returns false.
        assert!(
            !dict2.discard(wid_frequent, 0.001),
            "Should not discard frequent word with rand=0.001 < pdiscard"
        );
    }

    // =========================================================================
    // VAL-DICT-008: getLine word/label separation and OOV handling
    // =========================================================================

    #[test]
    fn test_getline_word_label_split() {
        // Build a dictionary with words and labels.
        let args = make_subword_args(0, 0, 0); // No subwords for simplicity
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("cat");
        dict.add("sit");
        dict.add("on");
        dict.add("mat");
        dict.add("__label__good");
        dict.add("__label__bad");
        dict.threshold(1, 1);
        // With maxn=0, init_ngrams just stores word IDs.
        dict.init_ngrams();

        let wid_cat = dict.get_id("cat");
        let wid_sit = dict.get_id("sit");
        let wid_on = dict.get_id("on");
        let wid_mat = dict.get_id("mat");
        let wid_good = dict.get_id("__label__good");
        let wid_bad = dict.get_id("__label__bad");
        let nwords = dict.nwords();

        // getLine from a text with words and a label.
        let text = "__label__good cat sit on mat\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;

        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        // Labels should contain the label ID (wid - nwords).
        assert_eq!(labels.len(), 1, "Should have one label");
        assert_eq!(
            labels[0],
            wid_good - nwords,
            "Label should be __label__good"
        );

        // Words should contain the word IDs.
        assert_eq!(words.len(), 4, "Should have 4 words");
        assert!(words.contains(&wid_cat), "Should contain 'cat'");
        assert!(words.contains(&wid_sit), "Should contain 'sit'");
        assert!(words.contains(&wid_on), "Should contain 'on'");
        assert!(words.contains(&wid_mat), "Should contain 'mat'");

        // Test with multiple labels.
        let text2 = "__label__good __label__bad cat sit\n";
        let mut reader2 = text2.as_bytes();
        let mut words2 = Vec::new();
        let mut labels2 = Vec::new();
        let mut pending2 = false;
        dict.get_line(&mut reader2, &mut words2, &mut labels2, &mut pending2);

        assert_eq!(labels2.len(), 2, "Should have two labels");
        assert!(labels2.contains(&(wid_good - nwords)));
        assert!(labels2.contains(&(wid_bad - nwords)));
        assert_eq!(words2.len(), 2);

        let _ = wid_bad; // suppress unused warning
    }

    #[test]
    fn test_getline_eos_terminates() {
        // EOS token terminates processing.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add(EOS);
        dict.add("hello");
        dict.add("world");
        dict.threshold(1, 1);
        dict.init_ngrams();

        // "hello world\nhello world" → first line: hello, world, EOS then stops
        let text = "hello world\nhello world\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;

        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        // EOS should terminate after first line (hello, world, then EOS).
        // words: hello, world (EOS is a word but it triggers break)
        // Actually looking at C++: addSubwords is called for EOS token if it's in vocab,
        // then "if (token == EOS) { break; }" → so the EOS word ID IS added to words.
        let wid_eos = dict.get_id(EOS);
        let wid_hello = dict.get_id("hello");
        let wid_world = dict.get_id("world");
        // EOS is added as a word, then we break.
        assert!(words.contains(&wid_eos), "words should contain EOS word id");
        assert!(words.contains(&wid_hello));
        assert!(words.contains(&wid_world));
        // The second line should not be included.
        assert_eq!(words.len(), 3, "Only first line + EOS should be in words");
    }

    #[test]
    fn test_getline_oov_with_subwords() {
        // OOV words with maxn>0: subwords computed on-the-fly.
        let args = make_subword_args(3, 6, 100000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello"); // in-vocab
        dict.threshold(1, 1);
        dict.init_ngrams();

        let text = "hello unknown\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;

        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        // "hello" is in-vocab → contributes word ID + n-grams.
        let wid_hello = dict.get_id("hello");
        // "unknown" is OOV → computed n-grams added (no word ID).
        // words should have subwords for hello + subwords for unknown.
        assert!(!words.is_empty(), "words should not be empty");
        // The word ID for "hello" should be in the list.
        assert!(
            words.contains(&wid_hello),
            "words should contain hello's word ID"
        );
        // OOV "unknown" contributes n-gram hashes (not word ID), they should be >= nwords.
        let nwords = dict.nwords();
        // All OOV-generated IDs should be >= nwords.
        for &id in &words {
            if id != wid_hello {
                // Could be a subword of hello or of unknown.
                // All n-gram bucket IDs are in [nwords, nwords+bucket).
                assert!(
                    id >= nwords,
                    "N-gram hash ID {} should be >= nwords {}",
                    id,
                    nwords
                );
            }
        }
    }

    #[test]
    fn test_getline_oov_without_subwords() {
        // OOV words with maxn=0: skipped entirely.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let text = "hello unknown_word\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;

        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        // Only "hello" should be in words; "unknown_word" is OOV and maxn=0 → skipped.
        let wid_hello = dict.get_id("hello");
        assert_eq!(words, vec![wid_hello], "Only 'hello' should be in words");
    }

    #[test]
    fn test_getline_oov_label_dropped() {
        // OOV labels are silently dropped.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.add("__label__good");
        dict.threshold(1, 1);
        dict.init_ngrams();

        // "__label__unknown" is OOV → dropped.
        let text = "__label__good __label__unknown hello\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;

        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        assert_eq!(labels.len(), 1, "Only known label should be in labels");
        let wid_good = dict.get_id("__label__good");
        let nwords = dict.nwords();
        assert_eq!(labels[0], wid_good - nwords);
    }

    #[test]
    fn test_getline_with_word_ngrams() {
        // Verify that word n-grams are added after processing the line.
        let mut args = Args::default();
        args.set_minn(0);
        args.set_maxn(0);
        args.set_bucket(100000);
        args.set_word_ngrams(2);
        let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);
        dict.add("hello");
        dict.add("world");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let text = "hello world\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;
        dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        let wid_hello = dict.get_id("hello");
        let wid_world = dict.get_id("world");
        let nwords = dict.nwords();

        // Words should contain: hello_id, world_id, plus one bigram hash.
        // (EOS is also in the line and gets its hash if it's in vocab, but it's not here)
        assert!(words.contains(&wid_hello), "words should contain hello ID");
        assert!(words.contains(&wid_world), "words should contain world ID");
        // The bigram hash should be appended.
        let h1 = crate::utils::hash(b"hello") as i32;
        let h2 = crate::utils::hash(b"world") as i32;
        let h_bigram = (h1 as i64 as u64)
            .wrapping_mul(116049371u64)
            .wrapping_add(h2 as i64 as u64);
        let bigram_id = nwords + (h_bigram % 100000) as i32;
        assert!(
            words.contains(&bigram_id),
            "words should contain bigram hash {}, words={:?}",
            bigram_id,
            words
        );
    }

    #[test]
    fn test_getline_returns_ntokens() {
        // ntokens should count all tokens including EOS.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add(EOS);
        dict.add("a");
        dict.add("b");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let text = "a b\n";
        let mut reader = text.as_bytes();
        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut pending = false;
        let ntokens = dict.get_line(&mut reader, &mut words, &mut labels, &mut pending);

        // Tokens: "a", "b", EOS → 3
        assert_eq!(ntokens, 3, "Should count 3 tokens (a, b, EOS)");
    }

    #[test]
    fn test_getline_from_str() {
        // get_line_from_str should separate words and labels, no EOS for newlines.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("cat");
        dict.add("mat");
        dict.add("__label__good");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let mut words = Vec::new();
        let mut labels = Vec::new();
        let ntokens = dict.get_line_from_str("__label__good cat mat", &mut words, &mut labels);

        assert_eq!(ntokens, 3);
        assert_eq!(labels.len(), 1);
        let wid_good = dict.get_id("__label__good");
        let nwords = dict.nwords();
        assert_eq!(labels[0], wid_good - nwords);
        assert_eq!(words.len(), 2);
    }

    // =========================================================================
    // Additional subword tests: get_subwords_for_string
    // =========================================================================

    #[test]
    fn test_get_subwords_for_string_in_vocab() {
        let args = make_subword_args(3, 6, 100000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let wid = dict.get_id("hello");
        let subwords_by_id = dict.get_subwords(wid).clone();
        let subwords_by_str = dict.get_subwords_for_string("hello");
        assert_eq!(
            subwords_by_id, subwords_by_str,
            "get_subwords_for_string should return same as get_subwords for in-vocab word"
        );
    }

    #[test]
    fn test_get_subwords_for_string_oov() {
        let args = make_subword_args(3, 6, 100000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        // OOV word: should compute n-grams without word ID.
        let oov_subwords = dict.get_subwords_for_string("unknown");
        // OOV does NOT prepend a word ID.
        let nwords = dict.nwords();
        for &id in &oov_subwords {
            assert!(
                id >= nwords,
                "OOV subword ID {} should be >= nwords {}",
                id,
                nwords
            );
        }
        assert!(
            !oov_subwords.is_empty(),
            "OOV word with maxn>0 should have subwords"
        );
    }

    #[test]
    fn test_get_subwords_for_string_eos_oov() {
        let args = make_subword_args(3, 6, 100000);
        let dict = Dictionary::new_with_capacity(args, 1024);
        // EOS as OOV should return empty.
        let eos_subwords = dict.get_subwords_for_string(EOS);
        assert!(eos_subwords.is_empty(), "EOS OOV should have no subwords");
    }

    #[test]
    fn test_add_subwords_oov_no_subwords_when_maxn_zero() {
        // add_subwords with OOV and maxn=0 → nothing added.
        let args = make_subword_args(0, 0, 0);
        let dict = Dictionary::new_with_capacity(args, 1024);
        let mut line = Vec::new();
        dict.add_subwords(&mut line, "oov_word", -1);
        assert!(line.is_empty(), "OOV with maxn=0 should add nothing");
    }

    #[test]
    fn test_add_subwords_in_vocab_with_subwords() {
        // add_subwords with in-vocab word and maxn>0 → full subwords vec.
        let args = make_subword_args(3, 6, 100000);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let wid = dict.get_id("hello");
        let mut line = Vec::new();
        dict.add_subwords(&mut line, "hello", wid);

        let expected = dict.get_subwords(wid).clone();
        assert_eq!(
            line, expected,
            "add_subwords should push all subwords for in-vocab word"
        );
    }

    #[test]
    fn test_add_subwords_in_vocab_no_subwords() {
        // add_subwords with in-vocab word and maxn=0 → only word ID.
        let args = make_subword_args(0, 0, 0);
        let mut dict = Dictionary::new_with_capacity(args, 1024);
        dict.add("hello");
        dict.threshold(1, 1);
        dict.init_ngrams();

        let wid = dict.get_id("hello");
        let mut line = Vec::new();
        dict.add_subwords(&mut line, "hello", wid);

        assert_eq!(
            line,
            vec![wid],
            "With maxn=0, add_subwords should only push word ID"
        );
    }
}
