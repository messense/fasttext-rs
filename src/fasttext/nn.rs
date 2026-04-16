use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::matrix::{DenseMatrix, Matrix};
use crate::utils::{self, OrdF32};
use crate::vector::Vector;

use super::FastText;

impl FastText {
    /// For each subword of the word (including the word itself if in vocab),
    /// returns the n-gram string and its corresponding vector from the input matrix.
    ///
    /// This mirrors the C++ `FastText::getNgramVectors`.
    pub fn get_ngram_vectors(&self, word: &str) -> Vec<(String, Vec<f32>)> {
        let dim = self.args.dim as usize;
        let entries = self.dict.get_ngram_strings(word);
        entries
            .into_iter()
            .map(|(id, s)| {
                let mut vec = vec![0.0f32; dim];
                if id >= 0 {
                    if self.quant {
                        // Quantized path: use QuantMatrix reconstruction.
                        if let Some(ref qi) = self.quant_input {
                            let mut v = Vector::new(dim);
                            qi.add_row_to_vector(&mut v, id, 1.0);
                            vec.copy_from_slice(v.data());
                        }
                    } else {
                        let row = self.input.row(id as i64);
                        vec.copy_from_slice(row);
                    }
                }
                (s, vec)
            })
            .collect()
    }

    /// Precompute L2-normalized word vectors for all vocabulary words.
    ///
    /// Returns a `DenseMatrix` where row `i` is the L2-normalized word vector
    /// for word `i`. Words with zero-norm vectors have a zero row.
    ///
    /// Used as a precomputation step for nearest-neighbor and analogy queries.
    pub fn precompute_word_vectors(&self) -> DenseMatrix {
        let nwords = self.dict.nwords() as usize;
        let dim = self.args.dim as i64;
        let dim_usize = dim as usize;
        let mut word_vectors = DenseMatrix::new(nwords as i64, dim);
        for i in 0..nwords {
            let word = self.dict.get_word(i as i32);
            let ids = self.dict.get_subwords_for_string(word);
            if ids.is_empty() {
                continue;
            }
            let row = word_vectors.row_mut(i as i64);
            let scale = 1.0 / ids.len() as f32;
            if self.quant {
                if let Some(ref qi) = self.quant_input {
                    let mut vec = Vector::new(dim_usize);
                    for &id in &ids {
                        qi.add_row_to_vector(&mut vec, id, scale);
                    }
                    row.copy_from_slice(vec.data());
                }
            } else {
                for &id in &ids {
                    let input_row = self.input.row(id as i64);
                    for (r, &v) in row.iter_mut().zip(input_row.iter()) {
                        *r += v * scale;
                    }
                }
            }
            utils::l2_normalize(row);
        }
        word_vectors
    }

    /// Find the `k` nearest neighbors to `word` by cosine similarity.
    ///
    /// Precomputes normalized word vectors for all vocabulary words, then
    /// linearly scans for the top-k words (excluding the query word itself).
    ///
    /// Returns a vec of `(similarity, word)` pairs sorted by descending similarity.
    ///
    /// This mirrors the C++ `FastText::getNN`.
    ///
    /// For repeated queries, use [`Self::precompute_word_vectors`] once and then
    /// call [`Self::get_nn_with_word_vectors`] to avoid recomputing on every call.
    pub fn get_nn(&self, word: &str, k: usize) -> Vec<(f32, String)> {
        let query = self.get_word_vector(word);
        let word_vectors = self.precompute_word_vectors();
        let ban_words = vec![word];
        self.nn_from_word_vectors(&word_vectors, &query, k, &ban_words)
    }

    /// Find the `k` nearest neighbors using precomputed word vectors.
    ///
    /// Like [`Self::get_nn`] but takes a precomputed `DenseMatrix` from
    /// [`Self::precompute_word_vectors`], avoiding the O(nwords × dim) recomputation
    /// on every call.
    pub fn get_nn_with_word_vectors(
        &self,
        word_vectors: &DenseMatrix,
        word: &str,
        k: usize,
    ) -> Vec<(f32, String)> {
        let query = self.get_word_vector(word);
        let ban_words = vec![word];
        self.nn_from_word_vectors(word_vectors, &query, k, &ban_words)
    }

    /// Find the `k` nearest neighbors to `word_a - word_b + word_c`.
    ///
    /// Computes the query vector as:
    ///   `query = normalize(A) - normalize(B) + normalize(C)`
    /// then finds the top-k nearest words (excluding A, B, C).
    ///
    /// Returns a vec of `(similarity, word)` pairs sorted by descending similarity.
    ///
    /// This mirrors the C++ `FastText::getAnalogies`.
    ///
    /// For repeated queries, use [`Self::precompute_word_vectors`] once and then
    /// call [`Self::get_analogies_with_word_vectors`] to avoid recomputing on every call.
    pub fn get_analogies(
        &self,
        word_a: &str,
        word_b: &str,
        word_c: &str,
        k: usize,
    ) -> Vec<(f32, String)> {
        let word_vectors = self.precompute_word_vectors();
        self.get_analogies_with_word_vectors(&word_vectors, word_a, word_b, word_c, k)
    }

    /// Find analogies using precomputed word vectors.
    ///
    /// Like [`Self::get_analogies`] but takes a precomputed `DenseMatrix` from
    /// [`Self::precompute_word_vectors`], avoiding the O(nwords × dim) recomputation
    /// on every call.
    pub fn get_analogies_with_word_vectors(
        &self,
        word_vectors: &DenseMatrix,
        word_a: &str,
        word_b: &str,
        word_c: &str,
        k: usize,
    ) -> Vec<(f32, String)> {
        let dim = self.args.dim as usize;
        let mut query = vec![0.0f32; dim];

        let buf = self.get_word_vector(word_a);
        let mut normalized = buf.clone();
        utils::l2_normalize(&mut normalized);
        for (q, &v) in query.iter_mut().zip(normalized.iter()) {
            *q += v;
        }

        let buf = self.get_word_vector(word_b);
        let mut normalized = buf.clone();
        utils::l2_normalize(&mut normalized);
        for (q, &v) in query.iter_mut().zip(normalized.iter()) {
            *q -= v;
        }

        let buf = self.get_word_vector(word_c);
        let mut normalized = buf.clone();
        utils::l2_normalize(&mut normalized);
        for (q, &v) in query.iter_mut().zip(normalized.iter()) {
            *q += v;
        }

        let ban_words = vec![word_a, word_b, word_c];
        self.nn_from_word_vectors(word_vectors, &query, k, &ban_words)
    }

    /// Internal: linear scan for top-k nearest neighbors given precomputed word vectors.
    ///
    /// `word_vectors` must be a matrix of L2-normalized word vectors (one per row).
    /// `query` is the raw (unnormalized) query vector.
    /// `ban_words` are excluded from the results.
    ///
    /// Returns a vec of `(similarity, word)` sorted by descending similarity.
    fn nn_from_word_vectors(
        &self,
        word_vectors: &DenseMatrix,
        query: &[f32],
        k: usize,
        ban_words: &[&str],
    ) -> Vec<(f32, String)> {
        if k == 0 {
            return Vec::new();
        }

        // Pre-normalize query so dot products directly give cosine similarity.
        let mut normalized_query: Vec<f32> = query.to_vec();
        utils::l2_normalize(&mut normalized_query);

        let nwords = self.dict.nwords() as usize;
        let dim = word_vectors.cols() as usize;
        let wv_data = word_vectors.data();
        let mut heap: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::with_capacity(k + 1);

        for i in 0..nwords {
            let word = self.dict.get_word(i as i32);
            if ban_words.contains(&word) {
                continue;
            }
            let row = &wv_data[i * dim..(i + 1) * dim];
            let sim: f32 = normalized_query
                .iter()
                .zip(row.iter())
                .map(|(&q, &r)| q * r)
                .sum();

            if heap.len() == k && sim <= heap.peek().unwrap().0 .0 .0 {
                continue;
            }
            heap.push(Reverse((OrdF32(sim), i)));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results: Vec<(f32, String)> = heap
            .into_iter()
            .map(|Reverse((OrdF32(sim), i))| (sim, self.dict.get_word(i as i32).to_string()))
            .collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}
