use std::{env, fs, str};

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
    cc::Build::new()
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
        .compile("cfasttext");
}

fn link_cpp() {
    // XXX: static link libc++?
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "macos" || target_os == "freebsd" {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target_os == "windows" {
        return;
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gcc");
    }
}

fn main() {
    fail_on_empty_directory("cfasttext");
    fail_on_empty_directory("cfasttext/fasttext");
    build_cfasttext();
    link_cpp();
}
