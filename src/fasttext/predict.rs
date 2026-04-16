use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::args::LossName;
use crate::dictionary::EOS;
use crate::error::{FastTextError, Result};
use crate::loss::find_k_best;
use crate::matrix::Matrix;
use crate::meter::Meter;
use crate::model::{Predictions, State};
use crate::utils;

use super::{FastText, Prediction};

impl FastText {
    /// Returns a list of `Prediction` pairs sorted by descending probability.
    /// Only predictions whose probability is ≥ `threshold` are returned.
    /// A negative `threshold` is treated as 0.
    ///
    /// Returns an empty vec for empty / whitespace-only input, `k = 0`, or
    /// models with no labels.
    pub fn predict(&self, text: &str, k: usize, threshold: f32) -> Vec<Prediction> {
        if k == 0 {
            return Vec::new();
        }
        // Negative threshold treated as 0
        let effective_threshold = threshold.max(0.0);

        // Tokenise into word (subword) IDs (no EOS from get_line_from_str).
        let mut words: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        self.dict.get_line_from_str(text, &mut words, &mut labels);
        if words.is_empty() {
            return Vec::new();
        }

        // Append EOS token to match C++ predictLine behavior: when reading
        // from a stream the newline character produces an EOS token that is
        // included in the hidden-representation average.
        let eos_id = self.dict.get_id(EOS);
        if let Some(eos_id) = eos_id {
            words.push(eos_id);
        }

        self.predict_words_internal(&words, k, effective_threshold)
    }

    /// Predict the top-`k` labels from pre-tokenized word IDs.
    ///
    /// `word_ids` must be valid input-matrix row indices (as produced by
    /// `Dictionary::get_line_from_str` or equivalent tokenization).
    ///
    /// # EOS handling
    ///
    /// Unlike [`predict`], this method does **not** automatically append the
    /// EOS token (`</s>`).  If you want results identical to `predict(text, …)`,
    /// you must append the EOS token ID yourself before calling this method:
    ///
    /// ```text
    /// let eos_id = model.dict().get_id("</s>");
    /// if let Some(eos_id) = eos_id { words.push(eos_id); }
    /// let preds = model.predict_on_words(&words, k, threshold);
    /// ```
    ///
    /// This design follows C++ fastText's `FastText::predictLine`, where the
    /// EOS token is injected by the stream tokenizer when a newline is read.
    /// The higher-level [`predict`] method replicates that behavior automatically
    /// (appending EOS after tokenizing the input string), while this lower-level
    /// method gives callers full control over the token sequence.
    pub fn predict_on_words(&self, word_ids: &[i32], k: usize, threshold: f32) -> Vec<Prediction> {
        if k == 0 || word_ids.is_empty() {
            return Vec::new();
        }
        let effective_threshold = threshold.max(0.0);
        self.predict_words_internal(word_ids, k, effective_threshold)
    }

    /// Internal helper: run model.predict on pre-validated word IDs.
    fn predict_words_internal(&self, word_ids: &[i32], k: usize, threshold: f32) -> Vec<Prediction> {
        let nlabels = self.dict.nlabels() as usize;
        if nlabels == 0 {
            return Vec::new();
        }

        let dim = self.args.dim as usize;
        let mut state = State::new(dim, nlabels, 0);

        // Clamp k to at most the number of labels.
        let k_eff = k.min(nlabels) as i32;

        let raw = if self.quant {
            self.predict_raw_quantized(word_ids, k_eff, threshold, &mut state)
        } else {
            self.model.predict(word_ids, k_eff, threshold, &mut state)
        };

        raw.into_iter()
            .map(|(log_prob, label_idx)| {
                let label = self
                    .dict
                    .get_label(label_idx)
                    .unwrap_or("__unknown__")
                    .to_string();
                Prediction {
                    prob: log_prob.exp(),
                    label,
                }
            })
            .collect()
    }

    /// Predict using quantized matrices (quant=true path).
    ///
    /// Computes hidden from `quant_input`, then computes output scores using
    /// `quant_output` (if present) or the dense `output` matrix.  Applies
    /// the appropriate normalization based on the configured loss function
    /// (softmax, OVA/sigmoid, or HS) and returns top-k predictions as
    /// (log_probability, label_index) pairs.
    fn predict_raw_quantized(
        &self,
        word_ids: &[i32],
        k: i32,
        threshold: f32,
        state: &mut State,
    ) -> Predictions {
        let quant_input = match self.quant_input.as_ref() {
            Some(qi) => qi,
            None => return Predictions::new(),
        };
        let nlabels = self.dict.nlabels() as usize;
        if nlabels == 0 || k <= 0 {
            return Predictions::new();
        }

        // Compute hidden representation using quantized input matrix.
        quant_input.average_rows_to_vector(&mut state.hidden, word_ids);

        // Compute raw output scores.
        let osz = nlabels;
        match &self.quant_output {
            Some(qout) => {
                for i in 0..osz {
                    let dot = qout.dot_row(&state.hidden, i as i64);
                    state.output[i] = dot;
                }
            }
            None => {
                // Use dense output matrix (qout=false).
                for i in 0..osz {
                    let dot = self.output.dot_row(&state.hidden, i as i64);
                    state.output[i] = dot;
                }
            }
        }

        // Apply the appropriate normalization based on loss type.
        match self.args.loss {
            LossName::OVA => {
                // One-vs-all: independent sigmoid per class.
                let tables = &self.loss_tables;
                for i in 0..osz {
                    state.output[i] = tables.sigmoid(state.output[i]);
                }
            }
            _ => {
                // Softmax (and NS, which also uses softmax for prediction).
                utils::softmax_in_place(state.output.data_mut(), osz);
            }
        }

        // Find top-k predictions above threshold.
        let mut heap = Predictions::new();
        find_k_best(k as usize, threshold, &mut heap, &state.output);
        heap
    }

    /// Evaluate the model on labeled test data and return a `Meter` with metrics.
    ///
    /// Reads each line from `reader`, extracts word IDs and gold label IDs via
    /// the dictionary, runs `k`-best prediction, and accumulates the results in
    /// a [`Meter`].  Lines with no labels or no words are skipped (matching C++
    /// `FastText::test`).
    ///
    /// # Arguments
    /// - `reader`: source of labeled test data (one labeled example per line).
    /// - `k`: number of top predictions to request per example.
    /// - `threshold`: minimum probability threshold (predictions below this
    ///   are excluded from the prediction set passed to the meter).
    ///
    /// Returns a [`Meter`] containing accumulated precision, recall, and F1
    /// statistics for all examples in `reader`.
    pub fn test_model<R: Read + Seek>(&self, reader: &mut R, k: usize, threshold: f32) -> Result<Meter> {
        let nlabels = self.dict.nlabels() as usize;
        let dim = self.args.dim as usize;
        let k_eff = if nlabels == 0 { 0i32 } else { k.min(nlabels) as i32 };
        let effective_threshold = threshold.max(0.0);

        if k_eff == 0 {
            return Ok(Meter::new());
        }

        // Rewind to the beginning, matching C++ `in.seekg(0, beg)`.
        reader.seek(SeekFrom::Start(0)).map_err(FastTextError::IoError)?;
        let mut buf_reader = BufReader::new(reader);

        let mut meter = Meter::new();
        let mut words: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut pending_newline = false;
        let mut word_hashes: Vec<i32> = Vec::new();
        let mut token = String::new();
        let mut state = State::new(dim, nlabels, 0);

        loop {
            let ntokens = self.dict.get_line_with_scratch(&mut buf_reader, &mut words, &mut labels, &mut word_hashes, &mut token, &mut pending_newline);
            if ntokens == 0 && words.is_empty() && labels.is_empty() {
                break;
            }

            if !labels.is_empty() && !words.is_empty() {
                let raw = if self.quant {
                    self.predict_raw_quantized(&words, k_eff, effective_threshold, &mut state)
                } else {
                    self.model.predict(&words, k_eff, effective_threshold, &mut state)
                };

                // Convert log-probs to probabilities (matching C++ `min(exp(score), 1.0)`).
                let predictions: Vec<(f32, i32)> = raw
                    .into_iter()
                    .map(|(log_prob, label_idx)| (log_prob.exp().min(1.0), label_idx))
                    .collect();

                meter.add(&predictions, &labels, k_eff as usize);
            }
        }

        Ok(meter)
    }
}
