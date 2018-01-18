extern crate cc;
extern crate cmake;

use std::fs;

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
    let dst = cmake::Config::new("cfasttext")
        .build_target("cfasttext_static")
        .build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=cfasttext_static");
}

fn main() {
    fail_on_empty_directory("cfasttext");
    fail_on_empty_directory("cfasttext/fasttext");
    build_cfasttext();
}
