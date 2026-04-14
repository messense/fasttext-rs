// Dictionary: tokenization, vocabulary, subwords, word n-grams, hash table

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use crate::args::{Args, ModelName};
use crate::error::{FastTextError, Result};
use crate::utils::hash;

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
            let is_ws =
                matches!(c, b' ' | b'\t' | b'\x0b' | b'\x0c' | b'\0' | b'\r' | b'\n');

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
        loop {
            match reader.read(&mut buf) {
                Ok(0) => {
                    // EOF: trigger eofbit equivalent, return what we have.
                    return !word.is_empty();
                }
                Ok(_) => {
                    let c = buf[0];
                    let is_ws = matches!(
                        c,
                        b' ' | b'\n' | b'\r' | b'\t' | b'\x0b' | b'\x0c' | b'\0'
                    );

                    if is_ws {
                        if word.is_empty() {
                            if c == b'\n' {
                                word.push_str(EOS);
                                return true;
                            }
                            // Skip non-newline whitespace when buffer is empty.
                        } else {
                            if c == b'\n' {
                                // Put back the newline via the pending flag.
                                *pending_newline = true;
                            }
                            return true;
                        }
                    } else {
                        // SAFETY: pushing raw bytes; caller should pass valid UTF-8.
                        word.push(c as char);
                    }
                }
                Err(_) => return !word.is_empty(),
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
            let found =
                Self::read_word_from_reader(reader, &mut pending_newline, &mut word);
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
            self.pdiscard[i] =
                (self.args.t() as f32 / f).sqrt() + self.args.t() as f32 / f;
        }
    }

    /// Call `init_table_discard()` publicly (used after binary load).
    pub fn init_discard(&mut self) {
        self.init_table_discard();
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
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::args::Args;

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
        assert_eq!(
            Dictionary::tokenize("hello world"),
            vec!["hello", "world"]
        );
        // Tab
        assert_eq!(
            Dictionary::tokenize("hello\tworld"),
            vec!["hello", "world"]
        );
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
        assert_eq!(
            Dictionary::tokenize("hello\0world"),
            vec!["hello", "world"]
        );
        // Carriage return (\r)
        assert_eq!(
            Dictionary::tokenize("hello\rworld"),
            vec!["hello", "world"]
        );
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
        assert_eq!(
            Dictionary::tokenize("a\nb"),
            vec!["a", EOS, "b"]
        );
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
        assert_eq!(result.len(), MAX_LINE_SIZE, "Expected {} tokens, got {}", MAX_LINE_SIZE, result.len());

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
        assert_ne!(hello_id, world_id, "hello and world should have different IDs");
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
        for _ in 0..5 { dict.add("rare"); }      // count 5
        for _ in 0..10 { dict.add("common"); }  // count 10
        for _ in 0..3 { dict.add("__label__cat"); } // label count 3
        for _ in 0..7 { dict.add("__label__dog"); } // label count 7

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
        dict.add("rare");       // count 1
        dict.add("common");     // count 2
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

        assert!(Dictionary::read_word_from_reader(&mut reader, &mut pending, &mut word));
        assert_eq!(word, "hello");

        assert!(Dictionary::read_word_from_reader(&mut reader, &mut pending, &mut word));
        assert_eq!(word, "world");

        // Newline was pending → EOS
        assert!(Dictionary::read_word_from_reader(&mut reader, &mut pending, &mut word));
        assert_eq!(word, EOS);

        assert!(Dictionary::read_word_from_reader(&mut reader, &mut pending, &mut word));
        assert_eq!(word, "foo");

        // EOF
        assert!(!Dictionary::read_word_from_reader(&mut reader, &mut pending, &mut word));
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
}
