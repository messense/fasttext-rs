// Model: compute_hidden, predict (top-k via min-heap), update (SGD)
//
// This module implements:
//   - `MinstdRng`: simple LCG matching C++ `std::minstd_rand`
//   - `State`: training/inference state (hidden, output, grad vectors + RNG)
//   - `Predictions`: type alias for the top-k result list
//   - `Model`: the core model struct with compute_hidden, predict, update

use std::sync::Arc;

use crate::loss::Loss;
use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

/// Top-k prediction results: list of (log_probability, label_index) pairs.
///
/// The pairs are stored in descending order by log_probability after a call
/// to `Loss::predict`.
pub type Predictions = Vec<(f32, i32)>;

// ============================================================================
// MinstdRng — matches C++ `std::minstd_rand`
// ============================================================================

/// Linear congruential random number generator matching C++ `std::minstd_rand`.
///
/// Parameters: multiplier=48271, increment=0, modulus=2^31-1=2147483647.
/// Sequence range: [1, 2147483646].
///
/// Used for negative sampling in `NegativeSamplingLoss`.
#[derive(Debug, Clone)]
pub struct MinstdRng {
    state: u64,
}

impl MinstdRng {
    /// Multiplier for the LCG.
    pub const A: u64 = 48271;
    /// Modulus for the LCG (2^31 - 1).
    pub const M: u64 = 2_147_483_647;

    /// Create a new RNG with the given seed.
    ///
    /// If `seed == 0`, the seed is set to 1 (matching C++ default_seed = 1).
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 1 } else { seed % Self::M };
        MinstdRng { state: if s == 0 { 1 } else { s } }
    }

    /// Advance the state and return the next value in [1, M-1].
    #[inline]
    pub fn generate(&mut self) -> u64 {
        self.state = (self.state * Self::A) % Self::M;
        self.state
    }

    /// Return a uniform value in `[0, n)`.
    #[inline]
    pub fn uniform_usize(&mut self, n: usize) -> usize {
        self.generate() as usize % n
    }
}

// ============================================================================
// State — per-example training/inference state
// ============================================================================

/// Per-example training and inference state.
///
/// Matches C++ `Model::State`: holds the intermediate computation vectors
/// (hidden layer, output layer, gradient) and the thread-local RNG.
#[derive(Debug)]
pub struct State {
    /// Hidden layer: average of input embedding rows.
    pub hidden: Vector,
    /// Output layer: scores or probabilities computed by the loss.
    pub output: Vector,
    /// Gradient: accumulated gradient to be distributed back to input embeddings.
    pub grad: Vector,
    /// Thread-local random number generator (used for negative sampling).
    pub rng: MinstdRng,

    /// Accumulated loss value (sum over all examples in this State).
    loss_value: f32,
    /// Number of examples processed so far.
    nexamples: i64,
}

impl State {
    /// Create a new State with zeroed vectors.
    ///
    /// - `hidden_size`: dimension of the hidden layer (= model `dim`)
    /// - `output_size`: number of output nodes (labels for supervised, words for unsupervised)
    /// - `seed`: initial seed for the internal RNG
    pub fn new(hidden_size: usize, output_size: usize, seed: u64) -> Self {
        State {
            hidden: Vector::new(hidden_size),
            output: Vector::new(output_size),
            grad: Vector::new(hidden_size),
            rng: MinstdRng::new(seed),
            loss_value: 0.0,
            nexamples: 0,
        }
    }

    /// Return the average loss per example so far.
    pub fn get_loss(&self) -> f32 {
        if self.nexamples == 0 {
            return 0.0;
        }
        self.loss_value / self.nexamples as f32
    }

    /// Record one example with the given loss value.
    pub fn increment_nexamples(&mut self, loss: f32) {
        self.nexamples += 1;
        self.loss_value += loss;
    }

    /// Return the number of examples processed so far.
    pub fn nexamples(&self) -> i64 {
        self.nexamples
    }

    /// Reset accumulated loss and example count.
    pub fn reset(&mut self) {
        self.loss_value = 0.0;
        self.nexamples = 0;
    }
}

// ============================================================================
// Model
// ============================================================================

/// Core fastText model: computes hidden representations, predicts top-k labels,
/// and performs SGD updates.
///
/// Matches C++ `class Model`.
pub struct Model {
    /// Input embedding matrix (word + subword vectors).
    ///
    /// Shared via `Arc` for Hogwild! training; mutation during `update` is done
    /// via a raw-pointer cast (matching C++ fastText's lock-free SGD).
    pub wi: Arc<DenseMatrix>,
    /// Loss function (holds the output weight matrix internally).
    pub loss: Box<dyn Loss>,
    /// Whether to normalize the hidden-layer gradient by `1 / |input|`.
    ///
    /// Set to `true` for supervised models, `false` for CBOW/skip-gram.
    pub normalize_gradient: bool,
    /// Model dimension (= `wi.cols()`).
    pub dim: usize,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("dim", &self.dim)
            .field("normalize_gradient", &self.normalize_gradient)
            .finish()
    }
}

impl Model {
    /// Sentinel value for "predict all labels" (matching C++ `kUnlimitedPredictions = -1`).
    ///
    /// **Note:** This constant is retained for documentation / API symmetry with C++.
    /// However, `Model::predict` treats *all* non-positive `k` (including this value)
    /// as "return nothing".  Callers that want all labels should pass
    /// `self.dict.nlabels() as i32` or use the higher-level `FastText::predict`
    /// API, which always clamps `k` to at most `nlabels`.
    pub const K_UNLIMITED_PREDICTIONS: i32 = -1;

    /// Construct a new `Model`.
    ///
    /// # Arguments
    /// * `wi` — shared input embedding matrix
    /// * `loss` — loss function (holds `wo` internally)
    /// * `normalize_gradient` — whether to normalize gradient by `1/|input|`
    ///   (use `true` for supervised, `false` for CBOW/skip-gram)
    pub fn new(wi: Arc<DenseMatrix>, loss: Box<dyn Loss>, normalize_gradient: bool) -> Self {
        let dim = wi.cols() as usize;
        Model { wi, loss, normalize_gradient, dim }
    }

    // -------------------------------------------------------------------------
    // compute_hidden
    // -------------------------------------------------------------------------

    /// Compute the hidden representation as the average of input-matrix rows.
    ///
    /// Sets `state.hidden = (1 / N) * Σ wi[input_ids[i]]`.
    /// For an empty `input_ids`, `state.hidden` is zeroed.
    ///
    /// Matches C++ `Model::computeHidden`.
    pub fn compute_hidden(&self, input_ids: &[i32], state: &mut State) {
        self.wi.average_rows_to_vector(&mut state.hidden, input_ids);
    }

    // -------------------------------------------------------------------------
    // predict
    // -------------------------------------------------------------------------

    /// Predict the top-`k` labels for the given input token IDs.
    ///
    /// Steps:
    /// 1. Compute the hidden representation.
    /// 2. Delegate to `loss.predict` which fills a min-heap and sorts descending.
    ///
    /// Returns an empty `Predictions` if `k <= 0` (including negative values) or
    /// `input_ids` is empty.
    ///
    /// Matches C++ `Model::predict`.
    pub fn predict(
        &self,
        input_ids: &[i32],
        k: i32,
        threshold: f32,
        state: &mut State,
    ) -> Predictions {
        let mut heap = Predictions::new();
        if k <= 0 || input_ids.is_empty() {
            return heap;
        }
        self.compute_hidden(input_ids, state);
        self.loss.predict(k, threshold, &mut heap, state);
        heap
    }

    // -------------------------------------------------------------------------
    // update
    // -------------------------------------------------------------------------

    /// Perform one SGD update step.
    ///
    /// Steps:
    /// 1. If `input_ids` is empty, return immediately (no update).
    /// 2. Compute the hidden representation.
    /// 3. Zero the gradient accumulator.
    /// 4. Forward pass through the loss (which also back-propagates the gradient
    ///    into `state.grad` and updates output weights in-place).
    /// 5. If `normalize_gradient`, scale `state.grad` by `1 / |input_ids|`.
    /// 6. For each input token ID, add `state.grad` to the corresponding input
    ///    embedding row (Hogwild!-style, lock-free).
    ///
    /// Matches C++ `Model::update`.
    pub fn update(
        &self,
        input_ids: &[i32],
        targets: &[i32],
        target_index: i32,
        lr: f32,
        state: &mut State,
    ) {
        if input_ids.is_empty() {
            return;
        }
        self.compute_hidden(input_ids, state);
        state.grad.zero();
        let loss_value = self.loss.forward(targets, target_index, state, lr, true);
        state.increment_nexamples(loss_value);
        if self.normalize_gradient {
            state.grad.mul(1.0 / input_ids.len() as f32);
        }
        // Hogwild!: unsynchronised write to wi is intentional (matching C++).
        // SAFETY: concurrent lock-free writes to different rows are benign by
        // the Hogwild! algorithm; same pattern used in loss.rs for wo.
        let wi_ptr = Arc::as_ptr(&self.wi) as *mut DenseMatrix;
        for &idx in input_ids {
            unsafe {
                (*wi_ptr).add_vector_to_row(&state.grad, idx as i64, 1.0);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── MinstdRng ──────────────────────────────────────────────────────────

    #[test]
    fn test_minstd_zero_seed_becomes_one() {
        let mut rng = MinstdRng::new(0);
        // First advance should be non-zero
        let v = rng.generate();
        assert!(v > 0, "First value should be positive");
        // With seed=1: generate = (1 * 48271) % 2147483647 = 48271
        let mut rng2 = MinstdRng::new(1);
        let v2 = rng2.generate();
        assert_eq!(v, v2, "seed=0 and seed=1 should produce same sequence");
    }

    #[test]
    fn test_minstd_sequence() {
        // Verify the LCG sequence matches expectations
        // With seed=1: state_n = (1 * 48271^n) mod (2^31-1)
        let mut rng = MinstdRng::new(1);
        let first = rng.generate(); // (1 * 48271) % 2147483647 = 48271
        assert_eq!(first, 48271);
        let second = rng.generate(); // (48271 * 48271) % 2147483647 = 48271^2 mod m
        assert_eq!(second, (48271u64 * 48271u64) % MinstdRng::M);
    }

    #[test]
    fn test_minstd_range() {
        let mut rng = MinstdRng::new(42);
        for _ in 0..1000 {
            let v = rng.generate();
            assert!(v >= 1 && v < MinstdRng::M, "Value out of range: {}", v);
        }
    }

    #[test]
    fn test_minstd_uniform_usize() {
        let mut rng = MinstdRng::new(7);
        let n = 100usize;
        for _ in 0..1000 {
            let v = rng.uniform_usize(n);
            assert!(v < n, "Uniform value {} should be < {}", v, n);
        }
    }

    // ── State ──────────────────────────────────────────────────────────────

    #[test]
    fn test_state_new_zeroed() {
        let s = State::new(10, 20, 0);
        assert_eq!(s.hidden.len(), 10);
        assert_eq!(s.output.len(), 20);
        assert_eq!(s.grad.len(), 10);
        for i in 0..10 {
            assert_eq!(s.hidden[i], 0.0);
            assert_eq!(s.grad[i], 0.0);
        }
        for i in 0..20 {
            assert_eq!(s.output[i], 0.0);
        }
    }

    #[test]
    fn test_state_get_loss_initial() {
        let s = State::new(4, 4, 0);
        assert_eq!(s.get_loss(), 0.0);
    }

    #[test]
    fn test_state_increment_nexamples() {
        let mut s = State::new(4, 4, 0);
        s.increment_nexamples(1.0);
        assert_eq!(s.get_loss(), 1.0);
        s.increment_nexamples(3.0);
        assert!((s.get_loss() - 2.0).abs() < 1e-6, "Expected 2.0, got {}", s.get_loss());
    }

    #[test]
    fn test_state_reset() {
        let mut s = State::new(4, 4, 0);
        s.increment_nexamples(5.0);
        s.reset();
        assert_eq!(s.get_loss(), 0.0);
    }

    // ── Model ─────────────────────────────────────────────────────────────

    use crate::loss::SoftmaxLoss;
    use crate::matrix::DenseMatrix;

    /// Build a `DenseMatrix` with the given `rows × cols` filled row-by-row.
    fn make_matrix(rows: i64, cols: i64, data: &[f32]) -> DenseMatrix {
        let mut m = DenseMatrix::new(rows, cols);
        m.data_mut().copy_from_slice(data);
        m
    }

    // ── compute_hidden ─────────────────────────────────────────────────────

    /// Average of rows [0, 1, 2] in a 3×4 matrix with known values.
    ///
    /// VAL-INF-006: `test_compute_hidden`
    #[test]
    fn test_compute_hidden_averaging() {
        // wi: 3 words × 4 dims
        // row 0: [1, 2, 3, 4]
        // row 1: [5, 6, 7, 8]
        // row 2: [9, 10, 11, 12]
        #[rustfmt::skip]
        let wi_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let wi = Arc::new(make_matrix(3, 4, &wi_data));
        let wo = Arc::new(DenseMatrix::new(2, 4)); // 2 labels
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);

        let mut state = State::new(4, 2, 0);
        let input_ids = [0i32, 1, 2];
        model.compute_hidden(&input_ids, &mut state);

        // Expected: mean of rows 0,1,2 = [(1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3]
        //         = [5, 6, 7, 8]
        let expected = [5.0f32, 6.0, 7.0, 8.0];
        for i in 0..4 {
            assert!(
                (state.hidden[i] - expected[i]).abs() < 1e-5,
                "hidden[{}] = {}, expected {}",
                i,
                state.hidden[i],
                expected[i]
            );
        }
    }

    /// Average of a single row.
    #[test]
    fn test_compute_hidden_single_row() {
        #[rustfmt::skip]
        let wi_data: Vec<f32> = vec![
            3.0, 1.0, 4.0, 1.0,
            5.0, 9.0, 2.0, 6.0,
        ];
        let wi = Arc::new(make_matrix(2, 4, &wi_data));
        let wo = Arc::new(DenseMatrix::new(2, 4));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);

        let mut state = State::new(4, 2, 0);
        model.compute_hidden(&[1i32], &mut state);

        let expected = [5.0f32, 9.0, 2.0, 6.0];
        for i in 0..4 {
            assert!(
                (state.hidden[i] - expected[i]).abs() < 1e-5,
                "hidden[{}] = {}, expected {}",
                i,
                state.hidden[i],
                expected[i]
            );
        }
    }

    /// Empty input should produce a zero hidden vector.
    ///
    /// VAL-INF-006: `test_compute_hidden_empty`
    #[test]
    fn test_compute_hidden_empty_input() {
        let wi = Arc::new(DenseMatrix::new(5, 4));
        let wo = Arc::new(DenseMatrix::new(3, 4));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);

        // Pre-fill hidden with non-zero values.
        let mut state = State::new(4, 3, 0);
        for i in 0..4 {
            state.hidden[i] = 99.0;
        }
        model.compute_hidden(&[], &mut state);

        for i in 0..4 {
            assert_eq!(state.hidden[i], 0.0, "hidden[{}] should be 0 for empty input", i);
        }
    }

    // ── predict ────────────────────────────────────────────────────────────

    /// `predict` returns top-k results sorted by descending log-probability.
    #[test]
    fn test_predict_returns_sorted_top_k() {
        // wi: 1 word × 3 dims  (hidden = the one word's row)
        let wi_data: Vec<f32> = vec![1.0, 0.0, 0.0];
        let wi = Arc::new(make_matrix(1, 3, &wi_data));

        // wo: 4 labels × 3 dims — rows designed so label 2 gets the highest score
        // label 0: wo[0] = [0, 0, 0] → dot = 0
        // label 1: wo[1] = [0.5, 0, 0] → dot = 0.5
        // label 2: wo[2] = [2.0, 0, 0] → dot = 2.0  (highest)
        // label 3: wo[3] = [1.0, 0, 0] → dot = 1.0
        #[rustfmt::skip]
        let wo_data: Vec<f32> = vec![
            0.0, 0.0, 0.0,
            0.5, 0.0, 0.0,
            2.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ];
        let wo = Arc::new(make_matrix(4, 3, &wo_data));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let mut state = State::new(3, 4, 0);
        let model = Model::new(Arc::clone(&wi), loss, false);

        let preds = model.predict(&[0i32], 2, 0.0, &mut state);

        assert_eq!(preds.len(), 2, "Should return 2 predictions");
        // Top prediction should be label 2 (highest dot product)
        assert_eq!(preds[0].1, 2, "First pred should be label 2, got {}", preds[0].1);
        // Second prediction should be label 3
        assert_eq!(preds[1].1, 3, "Second pred should be label 3, got {}", preds[1].1);
        // Sorted descending by log-prob
        assert!(
            preds[0].0 >= preds[1].0,
            "Predictions should be sorted descending: {:?}",
            preds
        );
    }

    /// `predict` returns empty vec for k=0.
    #[test]
    fn test_predict_k_zero_returns_empty() {
        let wi = Arc::new(DenseMatrix::new(3, 4));
        let wo = Arc::new(DenseMatrix::new(2, 4));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, false);
        let mut state = State::new(4, 2, 0);

        let preds = model.predict(&[0i32], 0, 0.0, &mut state);
        assert!(preds.is_empty(), "k=0 should return empty predictions");
    }

    /// `predict` returns empty vec for negative k (guard against non-positive k).
    ///
    /// Validates fix for the scrutiny finding: negative k values should return
    /// empty rather than causing unexpected behaviour (e.g., wrapping via `as usize`).
    #[test]
    fn test_predict_negative_k_returns_empty() {
        let wi = Arc::new(DenseMatrix::new(3, 4));
        let wo = Arc::new(DenseMatrix::new(2, 4));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, false);
        let mut state = State::new(4, 2, 0);

        for k in [-1i32, -2, -100, i32::MIN] {
            let preds = model.predict(&[0i32], k, 0.0, &mut state);
            assert!(
                preds.is_empty(),
                "k={} should return empty predictions",
                k
            );
        }
    }

    /// `predict` returns empty vec for empty input.
    #[test]
    fn test_predict_empty_input_returns_empty() {
        let wi = Arc::new(DenseMatrix::new(3, 4));
        let wo = Arc::new(DenseMatrix::new(2, 4));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, false);
        let mut state = State::new(4, 2, 0);

        let preds = model.predict(&[], 5, 0.0, &mut state);
        assert!(preds.is_empty(), "Empty input should return empty predictions");
    }

    // ── update ─────────────────────────────────────────────────────────────

    /// `update` with empty input should be a no-op.
    #[test]
    fn test_update_empty_input_noop() {
        #[rustfmt::skip]
        let wi_data: Vec<f32> = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let wi = Arc::new(make_matrix(2, 2, &wi_data.clone()));
        let wo = Arc::new(DenseMatrix::new(1, 2));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);
        let mut state = State::new(2, 1, 0);

        model.update(&[], &[0i32], 0, 0.1, &mut state);

        // wi should be unchanged
        assert_eq!(model.wi.data(), wi_data.as_slice());
    }

    /// `update` normalizes gradient by `1/|input|` for supervised (normalize=true).
    ///
    /// With a single input token, the gradient should NOT be scaled
    /// (1/1 = 1). With 2 tokens and `normalize=true` the gradient should be
    /// halved before being added to each input row.
    #[test]
    fn test_update_gradient_scaling_supervised() {
        // wi: 2 words × 2 dims  (all zeros so hidden = [0,0])
        let wi = Arc::new(DenseMatrix::new(2, 2));
        // wo: 2 labels × 2 dims
        // Set wo[0] = [1, 0] and wo[1] = [0, 1] so we get non-trivial gradients.
        let mut wo_mat = DenseMatrix::new(2, 2);
        wo_mat.data_mut()[0] = 1.0; // wo[0][0]
        wo_mat.data_mut()[3] = 1.0; // wo[1][1]
        let wo = Arc::new(wo_mat);

        // Test with normalize_gradient = true, 2 input tokens
        let loss_norm = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model_norm = Model::new(Arc::clone(&wi), loss_norm, true);
        let mut state_norm = State::new(2, 2, 0);
        model_norm.update(&[0i32, 1], &[0i32], 0, 0.1, &mut state_norm);
        let wi_norm_data: Vec<f32> = model_norm.wi.data().to_vec();

        // Test with normalize_gradient = false, same setup
        let wi2 = Arc::new(DenseMatrix::new(2, 2));
        let loss_no_norm = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model_no_norm = Model::new(Arc::clone(&wi2), loss_no_norm, false);
        let mut state_no_norm = State::new(2, 2, 0);
        model_no_norm.update(&[0i32, 1], &[0i32], 0, 0.1, &mut state_no_norm);
        let wi_no_norm_data: Vec<f32> = model_no_norm.wi.data().to_vec();

        // With normalize_gradient=true, gradient was multiplied by 1/2 before
        // being added to each row, so wi update should be smaller.
        // Specifically: delta(normalized) ≈ delta(unnormalized) / 2.
        for i in 0..4 {
            let delta_norm = wi_norm_data[i];    // started at 0
            let delta_no_norm = wi_no_norm_data[i];
            assert!(
                (delta_norm - delta_no_norm / 2.0).abs() < 1e-5,
                "Normalized gradient should be half the unnormalized at index {}: {} vs {}",
                i,
                delta_norm,
                delta_no_norm
            );
        }
    }

    /// `update` moves input weights in the correct direction.
    ///
    /// With hidden = [0, 0] (zero-weight word), wo[0]=[1,0], wo[1]=[0,1],
    /// target = label 0, the softmax gradient should be non-zero and the
    /// input embedding should be updated.
    #[test]
    fn test_update_weight_direction() {
        // wi: 1 word × 2 dims, all zeros → hidden = [0, 0]
        let wi = Arc::new(DenseMatrix::new(1, 2));
        // wo: 2 labels × 2 dims
        //   wo[0] = [1, 0], wo[1] = [0, 1]
        let mut wo_mat = DenseMatrix::new(2, 2);
        wo_mat.data_mut()[0] = 1.0; // wo[0][0]
        wo_mat.data_mut()[3] = 1.0; // wo[1][1]
        let wo = Arc::new(wo_mat);

        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);
        let mut state = State::new(2, 2, 0);

        // Record wi[0] before update
        let before: Vec<f32> = model.wi.row(0).to_vec();

        model.update(&[0i32], &[0i32], 0, 0.1, &mut state);

        // wi[0] should have changed:
        // hidden=[0,0], softmax → [0.5,0.5]
        // grad = lr*(1-0.5)*wo[0] + lr*(0-0.5)*wo[1]
        //      = 0.05*[1,0] + (-0.05)*[0,1] = [0.05, -0.05]
        // wi[0] += grad = [0.05, -0.05]
        let after: Vec<f32> = model.wi.row(0).to_vec();
        let changed = before.iter().zip(after.iter()).any(|(b, a)| (b - a).abs() > 1e-9);
        assert!(changed, "Input weights should be updated by SGD, before={:?} after={:?}", before, after);

        // Verify direction: wi[0][0] should increase (toward label 0 which has wo[0][0]=1)
        assert!(
            after[0] > before[0],
            "wi[0][0] should increase for target=label 0: {} -> {}",
            before[0],
            after[0]
        );
    }
}

