# fasttext-rs

[![CI](https://github.com/messense/fasttext-rs/actions/workflows/CI.yml/badge.svg)](https://github.com/messense/fasttext-rs/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/messense/fasttext-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/messense/fasttext-rs)
[![Crates.io](https://img.shields.io/crates/v/fasttext.svg)](https://crates.io/crates/fasttext)
[![docs.rs](https://docs.rs/fasttext/badge.svg)](https://docs.rs/fasttext/)

Pure Rust implementation of [fastText](https://github.com/facebookresearch/fastText)

## Installation

Add it to your `Cargo.toml`:

```toml
[dependencies]
fasttext = "0.7"
```

## Usage

### Training a model

```rust
use fasttext::FastText;
use fasttext::args::{Args, ModelName};

let mut args = Args::default();
args.input = "data.txt".to_string();
args.model = ModelName::Supervised;
args.epoch = 25;
args.lr = 1.0;

let model = FastText::train(args).unwrap();
model.save_model("model.bin").unwrap();
```

### Loading and predicting

```rust
use fasttext::FastText;

let model = FastText::load_model("model.bin").unwrap();
let predictions = model.predict("Which baking dish is best?", 3, 0.0);
for pred in &predictions {
    println!("{} {}", pred.label, pred.prob);
}
```

### Nearest neighbors

```rust
use fasttext::FastText;

let model = FastText::load_model("model.bin").unwrap();
let neighbors = model.get_nn("king", 10);
for (score, word) in &neighbors {
    println!("{word}\t{score}");
}
```

## CLI

A command-line tool compatible with the C++ fastText CLI is available behind the `cli` feature:

```bash
cargo install fasttext --features cli
```

It supports C++-style single-dash flags (e.g. `-epoch 25`) as well as standard double-dash flags (`--epoch 25`):

```bash
fasttext supervised -input data.txt -output model -epoch 25 -lr 1.0
fasttext predict model.bin test.txt
fasttext test model.bin test.txt
fasttext nn model.bin
```

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
