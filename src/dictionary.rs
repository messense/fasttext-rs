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

    // Hash table internals

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

    // Vocabulary management

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
    /// A word is a label iff it starts with `args.label` (default: `"__label__"`).
    pub fn get_type_from_str(&self, w: &str) -> EntryType {
        if w.starts_with(self.args.label.as_str()) {
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

    // Count accessors

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

    // Sorting and thresholding

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
            // Inline hash-table probing to avoid the `&self` borrow from `find_slot`.
            // Since `word2int` was just cleared to all -1, we only need to probe
            // until we find an empty slot (no word comparisons needed).
            let word_hash = hash(self.words[i].word.as_bytes());
            let len = self.word2int.len();
            let mut slot = (word_hash as usize) % len;
            while self.word2int[slot] != -1 {
                slot = (slot + 1) % len;
            }
            self.word2int[slot] = i as i32;
            self.size += 1;
            match self.words[i].entry_type {
                EntryType::Word => self.nwords += 1,
                EntryType::Label => self.nlabels += 1,
            }
        }
    }

    // Tokenization

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

    // Stream-based word reader (for readFromFile)

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

    // Vocabulary building from file

    /// Build vocabulary by reading tokens from a reader.
    ///
    /// Reads until EOF, incrementally thresholding if the vocabulary grows too large
    /// (> 75% of table capacity). After reading, applies final threshold using
    /// `args.min_count` and `args.min_count_label`, then computes the discard table.
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
            self.args.min_count as i64,
            self.args.min_count_label as i64,
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

    // Discard probability table

    /// Initialize the subsampling discard probability table.
    ///
    /// `pdiscard[i] = sqrt(t / f) + t / f` where `f = count / ntokens`.
    fn init_table_discard(&mut self) {
        self.pdiscard.resize(self.size as usize, 0.0);
        for i in 0..self.size as usize {
            let f = self.words[i].count as f32 / self.ntokens as f32;
            self.pdiscard[i] = (self.args.t as f32 / f).sqrt() + self.args.t as f32 / f;
        }
    }

    /// Call `init_table_discard()` publicly (used after binary load).
    pub fn init_discard(&mut self) {
        self.init_table_discard();
    }

    // Subword (n-gram) computation

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
        if self.args.bucket == 0 || self.pruneidx_size == 0 || id < 0 {
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
        let minn = self.args.minn;
        let maxn = self.args.maxn;
        let bucket = self.args.bucket;

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

    // Word n-gram hashing

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
        let bucket = self.args.bucket as u64;
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

    // Subword collection for a token (for getLine)

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
        } else if self.args.maxn <= 0 {
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
        let minn = self.args.minn;
        let maxn = self.args.maxn;
        let bucket = self.args.bucket;

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

    // getLine: read one line from a reader, separate words and labels

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

        self.add_word_ngrams(words, &word_hashes, self.args.word_ngrams);
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

        self.add_word_ngrams(words, &word_hashes, self.args.word_ngrams);
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
        if self.args.model == ModelName::SUP {
            return false;
        }
        rand > self.pdiscard[id as usize]
    }

    // Unsupervised getLine (for CBOW / skip-gram training)

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

    // Accessors for external use

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

    // Binary I/O

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

// Tests
