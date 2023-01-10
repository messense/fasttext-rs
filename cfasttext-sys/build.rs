use std::{fs, str};

fn fail_on_empty_directory(name: &str) {
    if fs::read_dir(name).unwrap().count() == 0 {
        println!(
            "The `{}` directory is empty, did you forget to pull the submodules?",
            name
        );
        println!("Try `git submodule update --init --recursive`");
        panic!();
    }
}

fn build_cfasttext() {
    let mut build = cc::Build::new();
    let compiler = build.get_compiler();
    if compiler.is_like_msvc() {
        // Enable exception for clang-cl
        // https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?redirectedfrom=MSDN&view=msvc-170
        build.flag("/EHsc");
    }
    build
        .cpp(true)
        .files([
            "cfasttext/fasttext/src/args.cc",
            "cfasttext/fasttext/src/autotune.cc",
            "cfasttext/fasttext/src/matrix.cc",
            "cfasttext/fasttext/src/dictionary.cc",
            "cfasttext/fasttext/src/loss.cc",
            "cfasttext/fasttext/src/productquantizer.cc",
            "cfasttext/fasttext/src/densematrix.cc",
            "cfasttext/fasttext/src/quantmatrix.cc",
            "cfasttext/fasttext/src/vector.cc",
            "cfasttext/fasttext/src/model.cc",
            "cfasttext/fasttext/src/utils.cc",
            "cfasttext/fasttext/src/meter.cc",
            "cfasttext/fasttext/src/fasttext.cc",
            "cfasttext/lib/cfasttext.cc",
        ])
        .includes(["cfasttext/fasttext/src", "cfasttext/include"])
        .flag("-std=c++11")
        .flag_if_supported("-pthread")
        .flag_if_supported("-funroll-loops")
        .compile("cfasttext_static");
}

fn main() {
    fail_on_empty_directory("cfasttext");
    fail_on_empty_directory("cfasttext/fasttext");
    build_cfasttext();
}
