// Benchmark: Rust pure implementation vs C++ (via cfasttext-sys FFI)
//
// Run with: cargo bench --bench bench_comparison

use std::ffi::CString;
use std::ptr;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cfasttext_sys::*;
use fasttext::FastText;

const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";
const COOKING_VALID: &str = "tests/fixtures/cooking.valid";

// Safe wrapper around the C++ fasttext handle for benchmarking.

struct CppFastText {
    handle: *mut fasttext_t,
    dim: i32,
}

impl CppFastText {
    fn load(path: &str) -> Self {
        unsafe {
            let handle = cft_fasttext_new();
            let c_path = CString::new(path).unwrap();
            let mut err: *mut std::os::raw::c_char = ptr::null_mut();
            cft_fasttext_load_model(handle, c_path.as_ptr(), &mut err);
            assert!(err.is_null(), "C++ load_model failed");
            let dim = cft_fasttext_get_dimension(handle);
            CppFastText { handle, dim }
        }
    }

    fn predict(&self, text: &str, k: i32, threshold: f32) {
        unsafe {
            let c_text = CString::new(text).unwrap();
            let mut err: *mut std::os::raw::c_char = ptr::null_mut();
            let preds = cft_fasttext_predict(self.handle, c_text.as_ptr(), k, threshold, &mut err);
            assert!(err.is_null(), "C++ predict failed");
            if !preds.is_null() {
                cft_fasttext_predictions_free(preds);
            }
        }
    }

    fn get_word_vector(&self, word: &str) {
        unsafe {
            let c_word = CString::new(word).unwrap();
            let mut buf = vec![0.0f32; self.dim as usize];
            cft_fasttext_get_word_vector(self.handle, c_word.as_ptr(), buf.as_mut_ptr());
            black_box(&buf);
        }
    }

    fn get_sentence_vector(&self, sentence: &str) {
        unsafe {
            let c_sentence = CString::new(sentence).unwrap();
            let mut buf = vec![0.0f32; self.dim as usize];
            cft_fasttext_get_sentence_vector(self.handle, c_sentence.as_ptr(), buf.as_mut_ptr());
            black_box(&buf);
        }
    }
}

impl Drop for CppFastText {
    fn drop(&mut self) {
        unsafe {
            cft_fasttext_free(self.handle);
        }
    }
}

fn bench_load_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_model");

    group.bench_function("rust", |b| {
        b.iter(|| {
            let model = FastText::load_model(black_box(COOKING_MODEL)).unwrap();
            black_box(&model);
        });
    });

    group.bench_function("cpp", |b| {
        b.iter(|| {
            let model = CppFastText::load(black_box(COOKING_MODEL));
            black_box(&model);
        });
    });

    group.finish();
}

fn bench_predict(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();
    let cpp_model = CppFastText::load(COOKING_MODEL);
    let input = "which baking dish is best to bake a banana bread ?";

    let mut group = c.benchmark_group("predict");

    for k in [1, 5] {
        group.bench_with_input(BenchmarkId::new("rust", k), &k, |b, &k| {
            b.iter(|| {
                let preds = rust_model.predict(black_box(input), k, 0.0);
                black_box(&preds);
            });
        });

        group.bench_with_input(BenchmarkId::new("cpp", k), &k, |b, &k| {
            b.iter(|| {
                cpp_model.predict(black_box(input), k as i32, 0.0);
            });
        });
    }

    group.finish();
}

fn bench_get_word_vector(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();
    let cpp_model = CppFastText::load(COOKING_MODEL);

    let mut group = c.benchmark_group("get_word_vector");

    group.bench_function("rust", |b| {
        b.iter(|| {
            let vec = rust_model.get_word_vector(black_box("banana"));
            black_box(&vec);
        });
    });

    group.bench_function("cpp", |b| {
        b.iter(|| {
            cpp_model.get_word_vector(black_box("banana"));
        });
    });

    group.finish();
}

fn bench_get_sentence_vector(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();
    let cpp_model = CppFastText::load(COOKING_MODEL);
    let sentence = "how to bake a banana bread";

    let mut group = c.benchmark_group("get_sentence_vector");

    group.bench_function("rust", |b| {
        b.iter(|| {
            let vec = rust_model.get_sentence_vector(black_box(sentence));
            black_box(&vec);
        });
    });

    group.bench_function("cpp", |b| {
        b.iter(|| {
            cpp_model.get_sentence_vector(black_box(sentence));
        });
    });

    group.finish();
}

fn bench_get_nn(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();

    let mut group = c.benchmark_group("get_nn");
    group.sample_size(10);

    group.bench_function("rust", |b| {
        b.iter(|| {
            let results = rust_model.get_nn(black_box("banana"), 5);
            black_box(&results);
        });
    });

    group.finish();
}

fn bench_get_analogies(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();

    let mut group = c.benchmark_group("get_analogies");
    group.sample_size(10);

    group.bench_function("rust", |b| {
        b.iter(|| {
            let results = rust_model.get_analogies(
                black_box("baking"),
                black_box("bread"),
                black_box("chicken"),
                5,
            );
            black_box(&results);
        });
    });

    group.finish();
}

fn bench_test_model(c: &mut Criterion) {
    let rust_model = FastText::load_model(COOKING_MODEL).unwrap();

    let mut group = c.benchmark_group("test_model");
    group.sample_size(10);

    for k in [1, 5] {
        group.bench_with_input(BenchmarkId::new("rust", k), &k, |b, &k| {
            b.iter(|| {
                let mut file = std::fs::File::open(COOKING_VALID).unwrap();
                let meter = rust_model.test_model(&mut file, k, 0.0).unwrap();
                black_box(&meter);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_load_model,
    bench_predict,
    bench_get_word_vector,
    bench_get_sentence_vector,
    bench_get_nn,
    bench_get_analogies,
    bench_test_model,
);
criterion_main!(benches);
