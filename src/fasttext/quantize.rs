use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64};
use std::sync::Arc;

use rayon::prelude::*;

use crate::args::{Args, ModelName};
use crate::dictionary::{EntryType, EOS};
use crate::error::{FastTextError, Result};
use crate::matrix::{DenseMatrix, Matrix};
use crate::model::Model;
use crate::quant_matrix::QuantMatrix;

use super::{build_loss, FastText, TrainThreadCtx};

impl FastText {
    /// Select the top `cutoff` embedding rows by L2 norm.
    ///
    /// The EOS token is always ranked first (so it is always retained).
    /// Remaining rows are sorted by descending L2 norm, and the top `cutoff`
    /// are returned.
    ///
    /// Matches C++ `FastText::selectEmbeddings`.
    fn select_embeddings(&self, cutoff: usize) -> Vec<i32> {
        let nrows = self.input.rows() as usize;
        let norms: Vec<f32> = (0..nrows)
            .map(|i| self.input.l2_norm_row(i as i64).unwrap_or(0.0))
            .collect();

        let eos_id = self.dict.get_id(EOS);

        let mut idx: Vec<i32> = (0..nrows as i32).collect();
        idx.sort_unstable_by(|&i1, &i2| {
            // EOS always comes first.
            if Some(i1) == eos_id && Some(i2) == eos_id {
                return std::cmp::Ordering::Equal;
            }
            if Some(i1) == eos_id {
                return std::cmp::Ordering::Less;
            }
            if Some(i2) == eos_id {
                return std::cmp::Ordering::Greater;
            }
            norms[i2 as usize]
                .partial_cmp(&norms[i1 as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let cutoff = cutoff.min(nrows);
        idx.truncate(cutoff);
        idx
    }

    /// Quantize the model in-place.
    ///
    /// Only supervised models can be quantized; attempting to quantize a CBOW or
    /// skip-gram model returns [`FastTextError::InvalidArgument`].
    ///
    /// Steps (matching C++ `FastText::quantize`):
    /// 1. Reject non-supervised models.
    /// 2. If `qargs.cutoff > 0` and smaller than current input rows:
    ///    a. Select top embeddings by L2 norm.
    ///    b. Prune the dictionary to those embeddings.
    ///    c. Build a new pruned input `DenseMatrix`.
    ///    d. If `qargs.retrain`: retrain with pruned input.
    /// 3. Create a `QuantMatrix` from the (possibly pruned) input.
    /// 4. If `qargs.qout`: create a `QuantMatrix` from the output.
    /// 5. Set `quant=true`, update `args.qout`.
    pub fn quantize(&mut self, qargs: &Args) -> Result<()> {
        if self.args.model != ModelName::Supervised {
            return Err(FastTextError::InvalidArgument(
                "For now we only support quantization of supervised models".to_string(),
            ));
        }

        // Copy input data to a mutable local buffer.
        let mut input_m = self.input.rows();
        let input_n = self.input.cols();
        let mut input_data: Vec<f32> = self.input.data().to_vec();

        let cutoff = qargs.cutoff;
        if cutoff > 0 && cutoff < input_m as usize {
            let idx = self.select_embeddings(cutoff);

            // Capture nwords before pruning to separate word rows from ngram rows.
            let nwords_before = self.dict.nwords();
            // Prune dictionary to the selected indices.
            // After prune(), words are kept in ascending original-ID order.
            self.dict.prune(&idx);

            // Separate word rows and ngram rows from `idx`.
            // Sort word rows ascending to match the pruned dictionary's word order.
            // Ngram rows keep their order from `idx` (which matches pruneidx mapping).
            let mut words_idx: Vec<i32> =
                idx.iter().copied().filter(|&i| i < nwords_before).collect();
            let ngrams_idx: Vec<i32> = idx
                .iter()
                .copied()
                .filter(|&i| i >= nwords_before)
                .collect();
            words_idx.sort_unstable(); // match dict word order (ascending original ID)

            // ordered_idx: word rows first (sorted), then ngram rows.
            // Invariant: matrix row j == dictionary word j (for j < new_nwords).
            let ordered_idx: Vec<i32> =
                words_idx.iter().chain(ngrams_idx.iter()).copied().collect();

            // Build pruned input matrix (rows in ordered_idx order).
            let pruned_m = ordered_idx.len() as i64;
            let mut pruned_data = vec![0f32; (pruned_m as usize) * (input_n as usize)];
            for (i, &old_row) in ordered_idx.iter().enumerate() {
                let src_start = old_row as usize * input_n as usize;
                let dst_start = i * input_n as usize;
                pruned_data[dst_start..dst_start + input_n as usize]
                    .copy_from_slice(&input_data[src_start..src_start + input_n as usize]);
            }
            input_m = pruned_m;
            input_data = pruned_data;

            if qargs.retrain {
                // Rebuild model with pruned input and retrain.
                let pruned_dense = DenseMatrix::from_data(pruned_m, input_n, &input_data);
                let pruned_arc = Arc::new(pruned_dense);
                self.retrain_after_prune(Arc::clone(&pruned_arc), qargs)?;
                // After retrain, use the (updated) pruned input data.
                input_data = pruned_arc.data().to_vec();
                // Update self.input so get_word_vector still works on dense models.
                self.input = pruned_arc;
            }
        }

        // Quantize the input matrix.
        let dsub = qargs.dsub as i32;
        let qnorm = qargs.qnorm;
        let quant_in = QuantMatrix::from_dense(&input_data, input_m, input_n, dsub, qnorm);
        self.quant_input = Some(quant_in);

        // Optionally quantize the output matrix.
        if qargs.qout {
            let output_data = self.output.data().to_vec();
            let out_m = self.output.rows();
            let out_n = self.output.cols();
            // C++ uses dsub=2 and passes the qnorm flag for the output matrix.
            let quant_out = QuantMatrix::from_dense(&output_data, out_m, out_n, 2, qnorm);
            self.quant_output = Some(quant_out);
        }

        // Mark model as quantized and update args.
        self.quant = true;
        let mut new_args = (*self.args).clone();
        new_args.qout = qargs.qout;
        self.args = Arc::new(new_args);

        Ok(())
    }

    /// Retrain the model after vocabulary pruning.
    ///
    /// Uses the `pruned_input` DenseMatrix (already compacted to the pruned
    /// vocabulary) and the existing output matrix.  Training hyperparameters
    /// (`epoch`, `lr`, `thread`) are taken from `qargs`.
    ///
    /// The pruned_input weights are updated in-place via Hogwild! SGD.
    fn retrain_after_prune(&self, pruned_input: Arc<DenseMatrix>, qargs: &Args) -> Result<()> {
        let input_path = &qargs.input;
        if input_path.is_empty() {
            return Err(FastTextError::InvalidArgument(
                "retrain=true requires qargs.input to be set to the training data path".to_string(),
            ));
        }

        // Build retrain args from qargs values.
        let mut retrain_args = (*self.args).clone();
        retrain_args.input = input_path.to_string();
        retrain_args.epoch = qargs.epoch;
        retrain_args.lr = qargs.lr;
        retrain_args.thread = qargs.thread;

        let n_threads = (retrain_args.thread as usize).max(1);
        let output_size = self.output.rows() as usize;
        let output = Arc::clone(&self.output);
        let target_counts = self.dict.get_counts(EntryType::Label);
        let normalize_gradient = true; // supervised

        let token_count = Arc::new(AtomicI64::new(0));
        let shared_loss = Arc::new(AtomicU64::new(f64::to_bits(0.0)));
        let shared_loss_count = Arc::new(AtomicI64::new(0));
        let abort_flag = Arc::new(AtomicBool::new(false));

        let training_results: Vec<Result<()>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_id| {
                let loss = build_loss(
                    &retrain_args,
                    Arc::clone(&output),
                    &target_counts,
                    Arc::clone(&self.loss_tables),
                );
                let model = Model::new(Arc::clone(&pruned_input), loss, normalize_gradient);
                let ctx = TrainThreadCtx {
                    args: &retrain_args,
                    dict: &self.dict,
                    model: &model,
                    output_size,
                    token_count: &token_count,
                    abort_flag: &abort_flag,
                    shared_loss: &shared_loss,
                    epoch_loss_tracker: None,
                };
                Self::train_thread_inner(thread_id, n_threads, &ctx, &shared_loss_count)
            })
            .collect();

        for r in training_results {
            r?;
        }

        Ok(())
    }
}
