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
    let dst = cmake::Config::new("cfasttext")
        .build_target("cfasttext_static")
        .build();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "windows" {
        let profile = match &*env::var("PROFILE").unwrap_or_else(|_| "debug".to_owned()) {
            "bench" | "release" => "Release",
            _ => "Debug",
        };
        println!(
            "cargo:rustc-link-search=native={}/build/{}",
            dst.display(),
            profile
        );
        println!(
            "cargo:rustc-link-search=native={}/build/fasttext/{}",
            dst.display(),
            profile
        );
    } else {
        println!("cargo:rustc-link-search=native={}/build", dst.display());
        println!(
            "cargo:rustc-link-search=native={}/build/fasttext",
            dst.display()
        );
    }
    println!("cargo:rustc-link-lib=static=cfasttext_static");
    println!("cargo:rustc-link-lib=static=fasttext");
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
