// Model: compute_hidden, predict (top-k via min-heap), update (SGD)
//
// This module defines the core types used by the loss functions:
//   - `MinstdRng`: simple LCG matching C++ `std::minstd_rand`
//   - `State`: training/inference state (hidden, output, grad vectors + RNG)
//   - `Predictions`: type alias for the top-k result list
//
// The `Model` struct itself (compute_hidden, predict, update) is implemented
// by the model-impl feature.

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

    /// Reset accumulated loss and example count.
    pub fn reset(&mut self) {
        self.loss_value = 0.0;
        self.nexamples = 0;
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
}
