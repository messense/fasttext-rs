// Dictionary tests: tokenization, vocabulary, subwords, word n-grams, getLine
//
// Extracted from src/dictionary.rs inline tests. These test the public
// API for Dictionary tokenization, vocabulary management, subword computation,
// word n-gram hashing, subsampling discard, and getLine word/label separation.

use std::sync::Arc;

use fasttext::args::{Args, ModelName};
use fasttext::dictionary::{Dictionary, EntryType, BOW, EOS, EOW, MAX_LINE_SIZE};

// Helpers

/// Create a default Args reference.
fn make_args() -> Arc<Args> {
    Arc::new(Args::default())
}

/// Create a small Dictionary for tests (capacity 1024 to avoid 120MB allocation).
fn make_dict() -> Dictionary {
    Dictionary::new_with_capacity(make_args(), 1024)
}

// VAL-DICT-001: Tokenization rules

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

// VAL-DICT-002: Label detection with configurable prefix

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
    args.label = "#".to_string();
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

// VAL-DICT-003: Vocabulary management

#[test]
fn test_vocab_lookup() {
    let mut dict = make_dict();
    dict.add("hello");
    dict.add("world");
    dict.add("hello");

    let hello_id = dict.get_id("hello");
    let world_id = dict.get_id("world");
    let oov_id = dict.get_id("foo");

    assert!(hello_id.is_some(), "hello should be in vocab");
    assert!(world_id.is_some(), "world should be in vocab");
    assert_ne!(
        hello_id, world_id,
        "hello and world should have different IDs"
    );
    assert_eq!(oov_id, None, "foo should be OOV");

    // Lookup by ID
    let hello_id = hello_id.unwrap();
    let world_id = world_id.unwrap();
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
    assert_eq!(dict.get_id("common"), Some(0));
    assert_eq!(dict.get_id("rare"), Some(1));
    assert_eq!(dict.get_id("__label__dog"), Some(2));
    assert_eq!(dict.get_id("__label__cat"), Some(3));
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

    assert!(dict.get_id("common").is_some(), "common should be in vocab");
    assert_eq!(dict.get_id("rare"), None, "rare should be filtered out");
    assert!(
        dict.get_id("__label__a").is_some(),
        "label a should be in vocab"
    );
    assert!(
        dict.get_id("__label__b").is_some(),
        "label b should be in vocab"
    );
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
        assert!(id.is_some(), "word_{} should be in vocab", i);
    }

    // OOV should return None
    assert_eq!(dict.get_id("not_in_vocab"), None);
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

// Additional edge case tests

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

    let id = dict.get_id("hello").unwrap();
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
    args.min_count = 1;
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

    assert!(id_jp.is_some(), "'日本語' should be in vocabulary");
    assert!(id_cafe.is_some(), "'café' should be in vocabulary");
    assert!(id_hello.is_some(), "'hello' should be in vocabulary");
    assert!(id_world.is_some(), "'world' should be in vocabulary");
    assert!(id_test.is_some(), "'test' should be in vocabulary");

    let id_jp = id_jp.unwrap();
    let id_cafe = id_cafe.unwrap();

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

    let eos_id = dict.get_id(EOS).unwrap();
    assert_eq!(dict.get_type_by_id(eos_id), EntryType::Word);
}

// VAL-DICT-004: Subword computation

/// Build an args with specific subword settings for testing.
fn make_subword_args(minn: i32, maxn: i32, bucket: i32) -> Arc<Args> {
    let mut args = Args::default();
    args.minn = minn;
    args.maxn = maxn;
    args.bucket = bucket;
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

    let wid = dict.get_id("hello").unwrap();
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
    let expected_hash_of_f_e_gt = (fasttext::utils::hash("fé>".as_bytes()) % 100000) as i32;
    assert!(
        ngrams.contains(&expected_hash_of_f_e_gt),
        "N-grams should contain hash('fé>') = {}",
        expected_hash_of_f_e_gt
    );
}

// VAL-DICT-005: Subword edge cases

#[test]
fn test_subword_eos_no_subwords() {
    // EOS should only have its own ID, no subword n-grams.
    let args = make_subword_args(3, 6, 2_000_000);
    let mut dict = Dictionary::new_with_capacity(args, 1024);
    dict.add(EOS);
    dict.add("hello");
    dict.threshold(1, 1);
    dict.init_ngrams();

    let eos_id = dict.get_id(EOS).unwrap();

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

    let wid = dict.get_id("hello").unwrap();
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

// VAL-DICT-006: Word n-gram hashing

#[test]
fn test_word_ngram_hash_bigram() {
    // Verify the rolling hash formula for a bigram.
    // h = hash(word1) (as i64 as u64), then h = h * 116049371 + hash(word2)
    // Result pushed is nwords + (h % bucket).
    let mut args = Args::default();
    args.bucket = 100000;
    args.word_ngrams = 2;
    let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    let h1 = fasttext::utils::hash(b"hello") as i32;
    let h2 = fasttext::utils::hash(b"world") as i32;
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
    args.bucket = 1_000_000;
    args.word_ngrams = 3;
    let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    let h1 = fasttext::utils::hash(b"the") as i32;
    let h2 = fasttext::utils::hash(b"quick") as i32;
    let h3 = fasttext::utils::hash(b"brown") as i32;
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
    args.bucket = 100000;
    args.word_ngrams = 1;
    let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    let hashes = vec![
        fasttext::utils::hash(b"hello") as i32,
        fasttext::utils::hash(b"world") as i32,
    ];
    let mut line = Vec::new();
    dict.add_word_ngrams(&mut line, &hashes, 1);

    assert!(line.is_empty(), "wordNgrams=1 should produce no n-grams");
}

#[test]
fn test_word_ngram_zero_bucket() {
    // With bucket=0, add_word_ngrams is a no-op.
    let mut args = Args::default();
    args.bucket = 0;
    args.word_ngrams = 2;
    let dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    let hashes = vec![
        fasttext::utils::hash(b"hello") as i32,
        fasttext::utils::hash(b"world") as i32,
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
    args.bucket = 500;
    args.word_ngrams = 3;
    let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);
    dict.add("a");
    dict.add("b");
    dict.add("c");
    dict.threshold(1, 1);

    let hashes: Vec<i32> = ["a", "b", "c"]
        .iter()
        .map(|w| fasttext::utils::hash(w.as_bytes()) as i32)
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

// VAL-DICT-007: Subsampling discard probability

#[test]
fn test_discard_table_formula() {
    // Build a dictionary with known frequencies and verify pdiscard formula.
    // pdiscard[i] = sqrt(t / f) + t / f  where f = count / ntokens.
    // With t=0.0001:
    //   If count=100, ntokens=1000 → f=0.1 → pdiscard = sqrt(0.001) + 0.001 ≈ 0.03262
    let mut args = Args::default();
    args.t = 0.0001;
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
    dict.init_discard();

    let wid = dict.get_id("word").unwrap();

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
    args.t = 1.0; // Very high threshold: would discard everything
    args.apply_supervised_defaults();
    let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    for _ in 0..3 {
        dict.add("word");
    }
    dict.threshold(1, 1);
    dict.init_discard();

    let wid = dict.get_id("word").unwrap();

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
    args.t = 0.9; // Very high t
    args.model = ModelName::SkipGram; // Unsupervised
    let mut dict = Dictionary::new_with_capacity(Arc::new(args), 1024);

    // Add "rare" once, "common" 1000 times → f_rare ≈ 0.001
    dict.add("rare");
    for _ in 0..1000 {
        dict.add("common");
    }
    dict.threshold(1, 1);
    dict.init_discard();

    let wid = dict.get_id("rare").unwrap();

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
    args2.t = 0.0001;
    args2.model = ModelName::SkipGram;
    let mut dict2 = Dictionary::new_with_capacity(Arc::new(args2), 1024);
    for _ in 0..9999 {
        dict2.add("frequent");
    }
    dict2.add("rare2");
    dict2.threshold(1, 1);
    dict2.init_discard();

    let wid_frequent = dict2.get_id("frequent").unwrap();
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

// VAL-DICT-008: getLine word/label separation and OOV handling

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

    let wid_cat = dict.get_id("cat").unwrap();
    let wid_sit = dict.get_id("sit").unwrap();
    let wid_on = dict.get_id("on").unwrap();
    let wid_mat = dict.get_id("mat").unwrap();
    let wid_good = dict.get_id("__label__good").unwrap();
    let wid_bad = dict.get_id("__label__bad").unwrap();
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
    let wid_eos = dict.get_id(EOS).unwrap();
    let wid_hello = dict.get_id("hello").unwrap();
    let wid_world = dict.get_id("world").unwrap();
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
    let wid_hello = dict.get_id("hello").unwrap();
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
    let wid_hello = dict.get_id("hello").unwrap();
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
    let wid_good = dict.get_id("__label__good").unwrap();
    let nwords = dict.nwords();
    assert_eq!(labels[0], wid_good - nwords);
}

#[test]
fn test_getline_with_word_ngrams() {
    // Verify that word n-grams are added after processing the line.
    let mut args = Args::default();
    args.minn = 0;
    args.maxn = 0;
    args.bucket = 100000;
    args.word_ngrams = 2;
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

    let wid_hello = dict.get_id("hello").unwrap();
    let wid_world = dict.get_id("world").unwrap();
    let nwords = dict.nwords();

    // Words should contain: hello_id, world_id, plus one bigram hash.
    // (EOS is also in the line and gets its hash if it's in vocab, but it's not here)
    assert!(words.contains(&wid_hello), "words should contain hello ID");
    assert!(words.contains(&wid_world), "words should contain world ID");
    // The bigram hash should be appended.
    let h1 = fasttext::utils::hash(b"hello") as i32;
    let h2 = fasttext::utils::hash(b"world") as i32;
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
    let wid_good = dict.get_id("__label__good").unwrap();
    let nwords = dict.nwords();
    assert_eq!(labels[0], wid_good - nwords);
    assert_eq!(words.len(), 2);
}

// Additional subword tests: get_subwords_for_string

#[test]
fn test_get_subwords_for_string_in_vocab() {
    let args = make_subword_args(3, 6, 100000);
    let mut dict = Dictionary::new_with_capacity(args, 1024);
    dict.add("hello");
    dict.threshold(1, 1);
    dict.init_ngrams();

    let wid = dict.get_id("hello").unwrap();
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

    let wid = dict.get_id("hello").unwrap();
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

    let wid = dict.get_id("hello").unwrap();
    let mut line = Vec::new();
    dict.add_subwords(&mut line, "hello", wid);

    assert_eq!(
        line,
        vec![wid],
        "With maxn=0, add_subwords should only push word ID"
    );
}
