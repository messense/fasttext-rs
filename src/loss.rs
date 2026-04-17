// Loss trait and implementations: Softmax, NegativeSampling, HierarchicalSoftmax, OneVsAll
//
// Matches C++ fastText `loss.cc / loss.h`.
//
// All loss types share:
//   - Precomputed sigmoid table (513 entries, range [-MAX_SIGMOID, MAX_SIGMOID])
//   - Precomputed log table    (513 entries, range (0, 1])
//   - Reference to the output weight matrix `wo` (shared via Arc)
//
// Loss hierarchy (mirrors C++):
//   Loss (trait)
//     ├─ SoftmaxLoss
//     └─ BinaryLogisticBase (shared helper struct)
//           ├─ OneVsAllLoss
//           ├─ NegativeSamplingLoss
//           └─ HierarchicalSoftmaxLoss

use std::sync::Arc;

use crate::matrix::{DenseMatrix, Matrix};
use crate::model::{MinstdRng, Predictions, State};
use crate::utils::{self, OrdF32};
use crate::vector::Vector;

// Constants

/// Number of entries in the sigmoid lookup table (SIGMOID_TABLE_SIZE).
pub const SIGMOID_TABLE_SIZE: i64 = 512;
/// Maximum absolute value of x for which sigmoid(x) is looked up in the table.
/// Values outside [-MAX_SIGMOID, MAX_SIGMOID] are saturated to 0 or 1.
pub const MAX_SIGMOID: i64 = 8;
/// Number of entries in the log lookup table (LOG_TABLE_SIZE).
pub const LOG_TABLE_SIZE: i64 = 512;
/// Size of the negative sampling table (NEGATIVE_TABLE_SIZE).
pub const NEGATIVE_TABLE_SIZE: i64 = 10_000_000;

// Helpers

/// `log(x + 1e-5)` — matches C++ `std_log` used in `findKBest` and HS DFS.
///
/// The small addend prevents log(0) = -∞.  C++ uses `1e-5` (double literal)
/// but the result is truncated to `real` (float); using `1e-5_f32` here
/// produces the same output since the difference is below f32 epsilon.
#[inline]
pub fn std_log(x: f32) -> f32 {
    (x + 1e-5_f32).ln()
}

// Precomputed tables

/// Precomputed sigmoid and log tables, shared by all loss implementations.
///
/// Both tables have `SIZE + 1` entries to allow index `SIZE` (from boundary
/// clipping).  The "size" constant refers to the number of intervals, not
/// the array length.
#[derive(Debug, Clone)]
pub struct LossTables {
    /// Sigmoid table: 513 entries, index i → sigmoid( i*2*MAX_SIGMOID/512 − MAX_SIGMOID ).
    sigmoid_table: Vec<f32>,
    /// Log table: 513 entries, index i → log( (i + 1e-5) / 512 ).
    log_table: Vec<f32>,
}

impl LossTables {
    /// Build the precomputed tables (called once per loss instance).
    pub fn new() -> Self {
        let sig_size = (SIGMOID_TABLE_SIZE + 1) as usize;
        let log_size = (LOG_TABLE_SIZE + 1) as usize;

        let sigmoid_table: Vec<f32> = (0..sig_size)
            .map(|i| {
                let x = (i as f32 * 2.0 * MAX_SIGMOID as f32) / SIGMOID_TABLE_SIZE as f32
                    - MAX_SIGMOID as f32;
                1.0 / (1.0 + (-x).exp())
            })
            .collect();

        let log_table: Vec<f32> = (0..log_size)
            .map(|i| {
                let x = (i as f32 + 1e-5_f32) / LOG_TABLE_SIZE as f32;
                x.ln()
            })
            .collect();

        LossTables {
            sigmoid_table,
            log_table,
        }
    }

    /// Fast sigmoid via lookup table.
    ///
    /// Saturates to 0.0 for x < -MAX_SIGMOID and to 1.0 for x > MAX_SIGMOID.
    #[inline]
    pub fn sigmoid(&self, x: f32) -> f32 {
        if x < -(MAX_SIGMOID as f32) {
            return 0.0;
        } else if x > MAX_SIGMOID as f32 {
            return 1.0;
        }
        let i = ((x + MAX_SIGMOID as f32) * SIGMOID_TABLE_SIZE as f32 / MAX_SIGMOID as f32 / 2.0)
            as i64;
        self.sigmoid_table[i as usize]
    }

    /// Fast natural log via lookup table.
    ///
    /// Input `x` must be in `(0, 1]`. Returns 0.0 for x > 1.0.
    #[inline]
    pub fn log(&self, x: f32) -> f32 {
        if x > 1.0 {
            return 0.0;
        }
        let i = (x * LOG_TABLE_SIZE as f32) as i64;
        self.log_table[i as usize]
    }
}

impl Default for LossTables {
    fn default() -> Self {
        Self::new()
    }
}

// Loss trait

/// Common interface for all fastText loss functions.
///
/// Matches C++ `class Loss`.
pub trait Loss: Send + Sync {
    /// Compute the loss for one training example and (optionally) update
    /// the output matrix gradient.
    ///
    /// # Arguments
    /// * `targets` — label indices for this example
    /// * `target_index` — which element of `targets` is the positive label
    /// * `state` — mutable per-thread training state
    /// * `lr` — current learning rate
    /// * `backprop` — if true, accumulate gradient into `state.grad` and
    ///   update the output matrix weights
    ///
    /// Returns the loss value (≥ 0).
    fn forward(
        &self,
        targets: &[i32],
        target_index: i32,
        state: &mut State,
        lr: f32,
        backprop: bool,
    ) -> f32;

    /// Write the output-layer scores / probabilities into `state.output`.
    fn compute_output(&self, state: &mut State);

    /// Fill `heap` with the top-`k` (log_prob, label_index) pairs whose
    /// probability exceeds `threshold`, sorted descending by log-prob.
    ///
    /// The default implementation calls `compute_output` then `find_k_best`.
    /// `HierarchicalSoftmaxLoss` overrides this with a DFS traversal.
    fn predict(&self, k: i32, threshold: f32, heap: &mut Predictions, state: &mut State) {
        self.compute_output(state);
        find_k_best(k as usize, threshold, heap, &state.output);
    }
}

// find_k_best — helper for the default predict implementation

/// Find the top-`k` entries in `output` that meet `threshold`, sorted
/// descending by log-probability.
///
/// Uses `std_log(output[i])` as the score stored in the heap, matching C++.
pub fn find_k_best(k: usize, threshold: f32, heap: &mut Predictions, output: &Vector) {
    if k == 0 {
        return;
    }
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Min-heap keyed on raw probability (smallest at top) via Reverse.
    // Since std_log is monotonically increasing, ordering by raw score
    // gives the same top-k as ordering by log-score. We only compute
    // std_log for the final k survivors, avoiding ln() on all other labels.
    let mut min_heap: BinaryHeap<Reverse<(OrdF32, i32)>> = BinaryHeap::with_capacity(k + 1);

    for (i, &score) in output.data().iter().enumerate() {
        if score < threshold {
            continue;
        }
        if min_heap.len() == k && score < min_heap.peek().unwrap().0 .0 .0 {
            continue;
        }
        min_heap.push(Reverse((OrdF32(score), i as i32)));
        if min_heap.len() > k {
            min_heap.pop();
        }
    }

    // Convert survivors to log-probability and sort descending.
    heap.extend(
        min_heap
            .into_iter()
            .map(|Reverse((OrdF32(s), idx))| (std_log(s), idx)),
    );
    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
}

// BinaryLogisticBase — shared helper for OVA, NS, HS

/// Shared binary logistic operations used by OneVsAll, NegativeSampling,
/// and HierarchicalSoftmax losses.
///
/// Matches C++ `BinaryLogisticLoss`.
#[derive(Debug)]
pub struct BinaryLogisticBase {
    tables: Arc<LossTables>,
    wo: Arc<DenseMatrix>,
}

impl BinaryLogisticBase {
    /// Create a new BinaryLogisticBase with shared precomputed tables.
    pub fn new(wo: Arc<DenseMatrix>, tables: Arc<LossTables>) -> Self {
        BinaryLogisticBase { tables, wo }
    }

    /// Compute the binary logistic loss for one node/target.
    ///
    /// score = sigmoid(wo[target] · hidden)
    /// loss  = -log(score) if labelIsPositive else -log(1 - score)
    /// grad  = lr * (label - score)   (if backprop)
    ///
    /// # Arguments
    /// * `target` — row index in the output matrix
    /// * `state` — mutable training state (hidden, grad)
    /// * `label_is_positive` — whether this is the positive (true) label
    /// * `lr` — learning rate
    /// * `backprop` — whether to accumulate gradient updates
    pub fn binary_logistic(
        &self,
        target: i32,
        state: &mut State,
        label_is_positive: bool,
        lr: f32,
        backprop: bool,
    ) -> f32 {
        let dot = self.wo.dot_row(&state.hidden, target as i64);
        let score = self.tables.sigmoid(dot);
        if backprop {
            let alpha = lr * (label_is_positive as i32 as f32 - score);
            // state.grad += alpha * wo[target]
            self.wo.add_row_to_vector(&mut state.grad, target, alpha);
            // wo[target] += alpha * hidden  (Hogwild! lock-free SGD)
            // SAFETY: see DenseMatrix::add_vector_to_row_unsync documentation.
            unsafe {
                self.wo
                    .add_vector_to_row_unsync(&state.hidden, target as i64, alpha);
            }
        }
        if label_is_positive {
            -self.tables.log(score)
        } else {
            -self.tables.log(1.0 - score)
        }
    }

    /// Compute output[i] = sigmoid(wo[i] · hidden) for all i.
    pub fn compute_output_sigmoid(&self, state: &mut State) {
        let osz = self.wo.rows() as usize;
        for (i, out) in state.output.data_mut()[..osz].iter_mut().enumerate() {
            let dot = self.wo.dot_row(&state.hidden, i as i64);
            *out = self.tables.sigmoid(dot);
        }
    }
}

// OneVsAllLoss

/// One-vs-all (OVA) loss: independent binary logistic for every class.
///
/// Matches C++ `OneVsAllLoss`.
#[derive(Debug)]
pub struct OneVsAllLoss {
    base: BinaryLogisticBase,
}

impl OneVsAllLoss {
    /// Create a new OVA loss with the given output weight matrix and shared tables.
    pub fn new(wo: Arc<DenseMatrix>, tables: Arc<LossTables>) -> Self {
        OneVsAllLoss {
            base: BinaryLogisticBase::new(wo, tables),
        }
    }
}

impl Loss for OneVsAllLoss {
    fn forward(
        &self,
        targets: &[i32],
        _target_index: i32,
        state: &mut State,
        lr: f32,
        backprop: bool,
    ) -> f32 {
        let osz = self.base.wo.rows() as usize;
        let target_set: std::collections::HashSet<i32> = targets.iter().copied().collect();
        (0..osz).fold(0.0f32, |loss, i| {
            let is_match = target_set.contains(&(i as i32));
            loss + self
                .base
                .binary_logistic(i as i32, state, is_match, lr, backprop)
        })
    }

    fn compute_output(&self, state: &mut State) {
        self.base.compute_output_sigmoid(state);
    }
}

// NegativeSamplingLoss

/// Negative sampling loss.
///
/// Matches C++ `NegativeSamplingLoss`.
#[derive(Debug)]
pub struct NegativeSamplingLoss {
    base: BinaryLogisticBase,
    /// Number of negative samples per positive example.
    neg: i32,
    /// Negative sampling table: ~10M entries, each is a label index.
    /// Distribution follows count^0.5 weighting.
    negatives: Vec<i32>,
}

impl NegativeSamplingLoss {
    /// Construct the negative sampling loss.
    ///
    /// Builds a table with distribution proportional to `count^0.5`, matching
    /// C++ `NegativeSamplingLoss::NegativeSamplingLoss`.  The table size will
    /// be approximately `NEGATIVE_TABLE_SIZE` (10M) depending on rounding.
    ///
    /// # Arguments
    /// * `wo` — shared output weight matrix
    /// * `neg` — number of negative samples per example
    /// * `target_counts` — label (or word) frequency counts
    pub fn new(
        wo: Arc<DenseMatrix>,
        neg: i32,
        target_counts: &[i64],
        tables: Arc<LossTables>,
    ) -> Self {
        let z: f64 = target_counts.iter().map(|&c| (c as f64).sqrt()).sum();
        let mut negatives: Vec<i32> = Vec::with_capacity(NEGATIVE_TABLE_SIZE as usize);
        for (i, &count) in target_counts.iter().enumerate() {
            let c = (count as f64).sqrt();
            let entries = (c * NEGATIVE_TABLE_SIZE as f64 / z) as usize;
            for _ in 0..entries {
                negatives.push(i as i32);
            }
        }
        NegativeSamplingLoss {
            base: BinaryLogisticBase::new(wo, tables),
            neg,
            negatives,
        }
    }

    /// Draw a negative sample that is different from `target`.
    ///
    /// Samples uniformly from the pre-built negative sampling table.  In the
    /// normal case the table contains many distinct label indices, so a different
    /// one is found quickly.
    ///
    /// # Infinite-loop protection (divergence from C++)
    ///
    /// C++ fastText uses an unbounded `do { … } while (target == negative)`
    /// loop which hangs on degenerate single-label tables.  This Rust
    /// implementation adds a `MAX_RETRIES` (100) guard: after 100 failed
    /// draws it returns `(target + 1) % n_labels` as a fallback.  This is a
    /// **known behavioral difference** — it prevents hangs at the cost of
    /// training on a deterministic (rather than random) negative sample in
    /// the degenerate case.
    fn get_negative(&self, target: i32, rng: &mut MinstdRng) -> i32 {
        const MAX_RETRIES: usize = 100;
        for _ in 0..MAX_RETRIES {
            let idx = rng.uniform_usize(self.negatives.len());
            let candidate = self.negatives[idx];
            if candidate != target {
                return candidate;
            }
        }
        // Degenerate fallback: return the label adjacent to `target` in the
        // output matrix, wrapping around so the index stays valid.
        let n_labels = self.base.wo.rows() as i32;
        if n_labels <= 1 {
            // Only one label exists; return 0 (same as target — unavoidable in
            // this degenerate case, but prevents an infinite loop).
            0
        } else {
            (target + 1) % n_labels
        }
    }

    /// Return a reference to the negative sampling table (for testing).
    #[cfg(test)]
    pub(crate) fn negatives(&self) -> &[i32] {
        &self.negatives
    }
}

impl Loss for NegativeSamplingLoss {
    fn forward(
        &self,
        targets: &[i32],
        target_index: i32,
        state: &mut State,
        lr: f32,
        backprop: bool,
    ) -> f32 {
        assert!(target_index >= 0);
        assert!((target_index as usize) < targets.len());
        let target = targets[target_index as usize];
        // Positive example
        let mut loss = self.base.binary_logistic(target, state, true, lr, backprop);
        // Negative examples
        for _ in 0..self.neg {
            let neg_target = self.get_negative(target, &mut state.rng);
            loss += self
                .base
                .binary_logistic(neg_target, state, false, lr, backprop);
        }
        loss
    }

    fn compute_output(&self, state: &mut State) {
        self.base.compute_output_sigmoid(state);
    }
}

// HierarchicalSoftmaxLoss

/// A single node in the Huffman tree.
#[derive(Debug, Clone)]
struct HsNode {
    parent: i32,
    left: i32,
    right: i32,
    count: i64,
    binary: bool,
}

/// Hierarchical softmax loss using a Huffman tree.
///
/// Matches C++ `HierarchicalSoftmaxLoss`.
#[derive(Debug)]
pub struct HierarchicalSoftmaxLoss {
    base: BinaryLogisticBase,
    /// Number of output nodes (labels).
    osz: i32,
    /// Path from each leaf to root: list of internal-node output-matrix row indices.
    paths: Vec<Vec<i32>>,
    /// Code for each leaf: which branch (left=false, right=true) was taken.
    codes: Vec<Vec<bool>>,
    /// The Huffman tree nodes (leaves: 0..osz-1, internals: osz..2*osz-2).
    tree: Vec<HsNode>,
}

impl HierarchicalSoftmaxLoss {
    /// Construct the HS loss and build the Huffman tree from `counts`.
    ///
    /// `counts` must be sorted in **descending** order (most frequent first),
    /// matching C++ fastText's label ordering.
    pub fn new(wo: Arc<DenseMatrix>, counts: &[i64], tables: Arc<LossTables>) -> Self {
        let osz = counts.len() as i32;
        let mut hs = HierarchicalSoftmaxLoss {
            base: BinaryLogisticBase::new(wo, tables),
            osz,
            paths: Vec::new(),
            codes: Vec::new(),
            tree: Vec::new(),
        };
        hs.build_tree(counts);
        hs
    }

    /// Build the Huffman tree from the given frequency counts.
    ///
    /// Algorithm: two-queue method.  Leaves (indices 0..osz-1) are processed
    /// from highest index to lowest (ascending count order when input is sorted
    /// descending).  Internal nodes (osz..2*osz-2) are built in order.
    fn build_tree(&mut self, counts: &[i64]) {
        self.build_huffman_nodes(counts);
        self.build_paths_and_codes();
    }

    /// Build the Huffman tree nodes from frequency counts.
    fn build_huffman_nodes(&mut self, counts: &[i64]) {
        let n = (2 * self.osz - 1) as usize;
        self.tree = vec![
            HsNode {
                parent: -1,
                left: -1,
                right: -1,
                count: 1_000_000_000_000_000_i64,
                binary: false,
            };
            n
        ];
        for (i, &c) in counts.iter().enumerate() {
            self.tree[i].count = c;
        }

        let mut leaf = self.osz - 1;
        let mut node = self.osz;

        for i in self.osz..(2 * self.osz - 1) {
            let i = i as usize;
            let mut mini = [0i32; 2];
            for mini_j in mini.iter_mut().take(2) {
                if leaf >= 0 && self.tree[leaf as usize].count < self.tree[node as usize].count {
                    *mini_j = leaf;
                    leaf -= 1;
                } else {
                    *mini_j = node;
                    node += 1;
                }
            }
            self.tree[i].left = mini[0];
            self.tree[i].right = mini[1];
            self.tree[i].count =
                self.tree[mini[0] as usize].count + self.tree[mini[1] as usize].count;
            self.tree[mini[0] as usize].parent = i as i32;
            self.tree[mini[1] as usize].parent = i as i32;
            self.tree[mini[1] as usize].binary = true;
        }
    }

    /// Build the root-to-leaf path and binary code for each leaf node.
    fn build_paths_and_codes(&mut self) {
        self.paths = vec![Vec::new(); self.osz as usize];
        self.codes = vec![Vec::new(); self.osz as usize];
        for i in 0..self.osz as usize {
            let mut j = i as i32;
            while self.tree[j as usize].parent != -1 {
                self.paths[i].push(self.tree[j as usize].parent - self.osz);
                self.codes[i].push(self.tree[j as usize].binary);
                j = self.tree[j as usize].parent;
            }
        }
    }

    /// Depth-first search through the Huffman tree to find top-k labels.
    ///
    /// Uses exact sigmoid (not the table) for score computation, matching C++.
    ///
    /// The heap is maintained in **descending** order by score (highest first),
    /// so the minimum element is at the end and can be evicted with `pop()` (O(1))
    /// instead of `remove(0)` (O(k)).
    fn dfs_with_hidden(
        &self,
        k: usize,
        threshold: f32,
        node: usize,
        score: f32,
        heap: &mut Predictions,
        hidden: &Vector,
    ) {
        let log_threshold = std_log(threshold);
        if score < log_threshold {
            return;
        }
        if heap.len() == k && !heap.is_empty() && score < heap.last().unwrap().0 {
            return;
        }

        let n = &self.tree[node];
        if n.left == -1 && n.right == -1 {
            // Leaf node: insert into heap maintaining descending sort order
            let pos = heap
                .iter()
                .position(|&(s, _)| s < score)
                .unwrap_or(heap.len());
            heap.insert(pos, (score, node as i32));
            if heap.len() > k {
                heap.pop(); // Remove the minimum (last element)
            }
            return;
        }

        // Internal node: exact sigmoid of wo[node - osz] · hidden
        let matrix_row = node as i32 - self.osz;
        let f_raw = self.base.wo.dot_row(hidden, matrix_row as i64);
        let f = 1.0_f32 / (1.0 + (-f_raw).exp()); // exact sigmoid

        let left = self.tree[node].left as usize;
        let right = self.tree[node].right as usize;

        self.dfs_with_hidden(k, threshold, left, score + std_log(1.0 - f), heap, hidden);
        self.dfs_with_hidden(k, threshold, right, score + std_log(f), heap, hidden);
    }

    /// Return the code length (tree depth) for leaf `i`.
    #[cfg(test)]
    pub(crate) fn depth(&self, i: usize) -> usize {
        self.codes[i].len()
    }

    /// Return the total number of nodes in the tree (leaves + internal).
    #[cfg(test)]
    pub(crate) fn tree_size(&self) -> usize {
        self.tree.len()
    }
}

impl Loss for HierarchicalSoftmaxLoss {
    fn forward(
        &self,
        targets: &[i32],
        target_index: i32,
        state: &mut State,
        lr: f32,
        backprop: bool,
    ) -> f32 {
        assert!(target_index >= 0);
        assert!((target_index as usize) < targets.len());
        let target = targets[target_index as usize] as usize;
        let path = &self.paths[target];
        let code = &self.codes[target];
        path.iter().zip(code.iter()).fold(0.0f32, |loss, (&p, &c)| {
            loss + self.base.binary_logistic(p, state, c, lr, backprop)
        })
    }

    fn compute_output(&self, state: &mut State) {
        // Compute approximate leaf probabilities by traversing each leaf's path.
        let osz = self.osz as usize;
        for (i, (path, code)) in self
            .paths
            .iter()
            .zip(self.codes.iter())
            .enumerate()
            .take(osz)
        {
            let log_prob = path
                .iter()
                .zip(code.iter())
                .fold(0.0f32, |acc, (&node, &c)| {
                    let dot = self.base.wo.dot_row(&state.hidden, node as i64);
                    let f = 1.0_f32 / (1.0 + (-dot).exp()); // exact sigmoid
                    if c {
                        acc + std_log(f)
                    } else {
                        acc + std_log(1.0 - f)
                    }
                });
            state.output[i] = log_prob.exp();
        }
    }

    fn predict(&self, k: i32, threshold: f32, heap: &mut Predictions, state: &mut State) {
        let root = (2 * self.osz - 2) as usize;
        let hidden = &state.hidden;
        self.dfs_with_hidden(k as usize, threshold, root, 0.0, heap, hidden);
        // Sort descending (highest log-prob first)
        heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }
}

// SoftmaxLoss

/// Full softmax loss with max-subtraction for numerical stability.
///
/// Matches C++ `SoftmaxLoss`.
#[derive(Debug)]
pub struct SoftmaxLoss {
    tables: Arc<LossTables>,
    wo: Arc<DenseMatrix>,
}

impl SoftmaxLoss {
    /// Create a new SoftmaxLoss with the given output weight matrix and shared tables.
    pub fn new(wo: Arc<DenseMatrix>, tables: Arc<LossTables>) -> Self {
        SoftmaxLoss { tables, wo }
    }
}

impl Loss for SoftmaxLoss {
    fn forward(
        &self,
        targets: &[i32],
        target_index: i32,
        state: &mut State,
        lr: f32,
        backprop: bool,
    ) -> f32 {
        self.compute_output(state);
        assert!(target_index >= 0);
        assert!((target_index as usize) < targets.len());
        let target = targets[target_index as usize];

        if backprop {
            let osz = self.wo.rows() as usize;
            for (i, &out_i) in state.output.data()[..osz].iter().enumerate() {
                let label = if i as i32 == target { 1.0_f32 } else { 0.0_f32 };
                let alpha = lr * (label - out_i);
                // state.grad += alpha * wo[i]
                self.wo.add_row_to_vector(&mut state.grad, i as i32, alpha);
                // wo[i] += alpha * hidden  (Hogwild! lock-free SGD)
                // SAFETY: see DenseMatrix::add_vector_to_row_unsync documentation.
                unsafe {
                    self.wo
                        .add_vector_to_row_unsync(&state.hidden, i as i64, alpha);
                }
            }
        }

        -self.tables.log(state.output[target as usize])
    }

    fn compute_output(&self, state: &mut State) {
        let osz = self.wo.rows() as usize;
        let cols = self.wo.cols() as usize;
        let hidden = state.hidden.data();
        let wo_data = self.wo.data();
        let out = state.output.data_mut();
        // Compute raw logits: output[i] = wo[i] · hidden
        for (i, out_i) in out[..osz].iter_mut().enumerate() {
            let row_start = i * cols;
            let row = &wo_data[row_start..row_start + cols];
            *out_i = row.iter().zip(hidden.iter()).map(|(&a, &b)| a * b).sum();
        }
        utils::softmax_in_place(&mut out[..osz]);
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // ── LossTables / sigmoid / log ─────────────────────────────────────────

    /// VAL-INF-001: sigmoid table size constant is SIGMOID_TABLE_SIZE (512)
    #[test]
    fn test_sigmoid_table_size() {
        assert_eq!(SIGMOID_TABLE_SIZE, 512);
        let tables = LossTables::new();
        // C++ stores SIGMOID_TABLE_SIZE + 1 entries
        assert_eq!(
            tables.sigmoid_table.len(),
            (SIGMOID_TABLE_SIZE + 1) as usize
        );
    }

    /// VAL-INF-001: log table size constant is LOG_TABLE_SIZE (512)
    #[test]
    fn test_log_table_size() {
        assert_eq!(LOG_TABLE_SIZE, 512);
        let tables = LossTables::new();
        assert_eq!(tables.log_table.len(), (LOG_TABLE_SIZE + 1) as usize);
    }

    /// VAL-INF-001: sigmoid boundary values
    #[test]
    fn test_sigmoid_boundary_values() {
        let tables = LossTables::new();
        // sigmoid(-8) ≈ 0 (from table, at the boundary)
        let s_neg = tables.sigmoid(-(MAX_SIGMOID as f32));
        assert!(s_neg < 0.01, "sigmoid(-8) should be near 0, got {}", s_neg);
        // sigmoid(0) ≈ 0.5
        let s_zero = tables.sigmoid(0.0);
        assert!(
            (s_zero - 0.5).abs() < 1e-3,
            "sigmoid(0) should be ~0.5, got {}",
            s_zero
        );
        // sigmoid(8) ≈ 1 (from table, at the boundary)
        let s_pos = tables.sigmoid(MAX_SIGMOID as f32);
        assert!(s_pos > 0.99, "sigmoid(8) should be near 1, got {}", s_pos);
        // Outside range: full saturation
        assert_eq!(tables.sigmoid(-100.0), 0.0);
        assert_eq!(tables.sigmoid(100.0), 1.0);
    }

    /// VAL-INF-001: sigmoid table accuracy vs true sigmoid within 1e-3
    #[test]
    fn test_sigmoid_accuracy() {
        let tables = LossTables::new();
        for &x in &[-7.5f32, -4.0, -1.0, 0.0, 1.0, 4.0, 7.5] {
            let approx = tables.sigmoid(x);
            let exact = 1.0 / (1.0 + (-x).exp());
            assert!(
                (approx - exact).abs() < 1e-3,
                "sigmoid({}) approx={} exact={} diff={}",
                x,
                approx,
                exact,
                (approx - exact).abs()
            );
        }
    }

    /// VAL-INF-001: log table accuracy
    ///
    /// The log table has 512 intervals over (0, 1].  At a given x, the
    /// quantization error is bounded by |ln(x + 1/512) - ln(x)| ≈ 1/(512·x).
    /// For x ≥ 0.1 this is < 0.02; for very small x the error grows.
    /// We test representative values in the range where the table is accurate.
    #[test]
    fn test_log_table_accuracy() {
        let tables = LossTables::new();
        // Test mid-range values where table is reasonably accurate
        for &x in &[0.1f32, 0.25, 0.5, 0.75, 1.0] {
            let approx = tables.log(x);
            let exact = x.ln();
            // Expected error bound: 1/(512*x) ≈ 0.02 for x=0.1
            assert!(
                (approx - exact).abs() < 0.03,
                "log({}) approx={} exact={} diff={}",
                x,
                approx,
                exact,
                (approx - exact).abs()
            );
        }
        // log(1) should be very close to 0 (boundary value per VAL-INF-001)
        assert!(tables.log(1.0).abs() < 0.01, "log(1) should be near 0");
        // log(>1) should return 0.0
        assert_eq!(tables.log(1.5), 0.0);
    }

    /// VAL-INF-001: log(small positive) should be large negative
    #[test]
    fn test_log_small_positive() {
        let tables = LossTables::new();
        // Very small positive x → large negative log
        let val = tables.log(0.001);
        assert!(val < -4.0, "log(0.001) should be < -4, got {}", val);
    }

    // ── NegativeSamplingLoss ───────────────────────────────────────────────

    /// VAL-INF-002: negative table has approximately 10M entries
    ///
    /// The table uses floating-point proportions so may be slightly under 10M
    /// due to integer truncation, matching C++ behaviour.
    #[test]
    fn test_ns_table_size() {
        let wo = Arc::new(DenseMatrix::new(4, 10));
        let counts = vec![100i64, 200, 50, 150];
        let loss = NegativeSamplingLoss::new(wo, 5, &counts, Arc::new(LossTables::new()));
        let size = loss.negatives().len();
        // Allow up to 100 entries of slack from floating-point truncation
        assert!(
            size >= (NEGATIVE_TABLE_SIZE - 100) as usize && size <= NEGATIVE_TABLE_SIZE as usize,
            "NS negative table size {} should be within 100 entries of 10M ({})",
            size,
            NEGATIVE_TABLE_SIZE
        );
    }

    /// VAL-INF-002: negative table distribution follows count^0.5
    #[test]
    fn test_ns_distribution() {
        let wo = Arc::new(DenseMatrix::new(4, 10));
        // counts = [4, 1]: sqrt ratios = [2, 1], so label 0 should have ~2/3 of entries
        let counts = vec![4i64, 1];
        let loss = NegativeSamplingLoss::new(wo, 5, &counts, Arc::new(LossTables::new()));
        let negs = loss.negatives();
        let count_0 = negs.iter().filter(|&&x| x == 0).count();
        let count_1 = negs.iter().filter(|&&x| x == 1).count();
        let total = (count_0 + count_1) as f64;
        let ratio = count_0 as f64 / total;
        // Expected ratio: sqrt(4) / (sqrt(4) + sqrt(1)) = 2/3 ≈ 0.667
        assert!(
            (ratio - 2.0 / 3.0).abs() < 0.05,
            "NS distribution ratio {} should be ~0.667 (2/3)",
            ratio
        );
    }

    /// VAL-INF-002: NegativeSamplingLoss forward returns a finite, non-negative value
    #[test]
    fn test_ns_forward() {
        let dim = 4usize;
        let nlabels = 3usize;
        let wo = Arc::new(DenseMatrix::new(nlabels as i64, dim as i64));
        let counts = vec![100i64, 50, 25];
        let loss = NegativeSamplingLoss::new(wo, 2, &counts, Arc::new(LossTables::new()));

        let mut state = State::new(dim, nlabels, 42);
        state.hidden[0] = 1.0;
        state.hidden[1] = 0.5;

        let targets = vec![0i32];
        let result = loss.forward(&targets, 0, &mut state, 0.1, false);
        assert!(
            result.is_finite(),
            "NS forward should return finite value, got {}",
            result
        );
        assert!(result >= 0.0, "NS forward should return non-negative loss");
    }

    // ── HierarchicalSoftmaxLoss ────────────────────────────────────────────

    /// VAL-INF-003: Huffman tree — more frequent targets get shorter codes
    #[test]
    fn test_hs_huffman_tree_depth_ordering() {
        let osz = 5usize;
        // internal nodes = osz - 1 = 4 rows in wo
        let wo = Arc::new(DenseMatrix::new((osz - 1) as i64, 10));
        // Counts sorted DESCENDING: most frequent first
        let counts = vec![100i64, 80, 60, 20, 10];
        let loss = HierarchicalSoftmaxLoss::new(wo, &counts, Arc::new(LossTables::new()));

        // Tree should have 2*osz - 1 nodes
        assert_eq!(loss.tree_size(), 2 * osz - 1);
        // Most frequent label (index 0, count=100) should have the shortest code
        let depth_0 = loss.depth(0);
        let depth_4 = loss.depth(4); // least frequent
        assert!(
            depth_0 <= depth_4,
            "Most frequent label (depth={}) should have code length <= least frequent (depth={})",
            depth_0,
            depth_4
        );
    }

    /// VAL-INF-003: Huffman tree with 2 labels — each gets path length 1
    #[test]
    fn test_hs_huffman_tree_two_labels() {
        let osz = 2usize;
        let wo = Arc::new(DenseMatrix::new((osz - 1) as i64, 4));
        let counts = vec![100i64, 50];
        let loss = HierarchicalSoftmaxLoss::new(wo, &counts, Arc::new(LossTables::new()));

        // With 2 labels, each should have path length 1
        assert_eq!(loss.depth(0), 1);
        assert_eq!(loss.depth(1), 1);
        // Tree has 2*2-1 = 3 nodes
        assert_eq!(loss.tree_size(), 3);
    }

    /// VAL-INF-003: Huffman tree — forward pass returns finite, non-negative value
    #[test]
    fn test_hs_forward() {
        let osz = 4usize;
        let dim = 8usize;
        // internal nodes = osz - 1 = 3 rows in wo
        let wo = Arc::new(DenseMatrix::new((osz - 1) as i64, dim as i64));
        let counts = vec![100i64, 80, 40, 20];
        let loss = HierarchicalSoftmaxLoss::new(wo, &counts, Arc::new(LossTables::new()));

        let mut state = State::new(dim, osz, 1);
        state.hidden[0] = 1.0;

        let targets = vec![2i32];
        let result = loss.forward(&targets, 0, &mut state, 0.05, false);
        assert!(
            result.is_finite(),
            "HS forward should be finite, got {}",
            result
        );
        assert!(
            result >= 0.0,
            "HS forward should be non-negative, got {}",
            result
        );
    }

    /// VAL-INF-003: HierarchicalSoftmaxLoss DFS prediction returns correct top label
    ///
    /// Tree for counts=[100, 80, 40, 20] (osz=4):
    ///   leaf 0 (count=100): depth=1 (shortest code, left child of root)
    ///   leaf 1 (count=80):  depth=2 (right of internal node 5, which is right of root)
    ///   leaves 2,3 (count=40,20): depth=3
    ///
    /// By setting wo[2][0]=-10 (the root's output-matrix row) with hidden=[1,0,0,0]:
    ///   sigmoid(-10) ≈ 0 → left branch (to leaf 0) gets score ≈ std_log(1) ≈ 0
    ///   right branch gets score ≈ std_log(0) ≈ -11.5 (very negative)
    /// So label 0 should be the top-1 prediction.
    #[test]
    fn test_hs_dfs_prediction() {
        let osz = 4usize;
        let dim = 4usize;
        // internal nodes = osz - 1 = 3 rows in wo
        let mut wo = DenseMatrix::new((osz - 1) as i64, dim as i64);
        // counts sorted DESCENDING: most frequent first
        let counts = vec![100i64, 80, 40, 20];

        // wo[2][0] = -10 → at root (node 6=2*4-2) matrix_row=6-4=2
        // sigmoid(wo[2]·hidden) = sigmoid(-10*1) ≈ 0
        // Left child (leaf 0): score ≈ std_log(1-0) ≈ 0 (high)
        // Right child (node 5): score ≈ std_log(0) ≈ -11.5 (low)
        *wo.at_mut(2, 0) = -10.0;
        let wo = Arc::new(wo);
        let loss = HierarchicalSoftmaxLoss::new(wo, &counts, Arc::new(LossTables::new()));

        let mut state = State::new(dim, osz, 0);
        state.hidden[0] = 1.0; // unit vector along dim 0

        let mut heap = Predictions::new();
        loss.predict(1, 0.0, &mut heap, &mut state);

        assert_eq!(heap.len(), 1, "HS DFS predict(k=1) should return 1 result");
        assert_eq!(
            heap[0].1, 0,
            "Most frequent label (index 0, depth=1) should be top-1 prediction, got {}",
            heap[0].1
        );
        assert!(
            heap[0].0.is_finite(),
            "Log-prob should be finite, got {}",
            heap[0].0
        );

        // threshold=0.01 → log_threshold = std_log(0.01) = ln(0.01+1e-5) ≈ -4.6
        // Right branch score ≈ -11.5 < -4.6 → pruned
        // Left branch (label 0) score ≈ 0 > -4.6 → survives
        let mut heap2 = Predictions::new();
        loss.predict(4, 0.01, &mut heap2, &mut state);

        // With pruning, only label 0 should survive
        assert!(
            !heap2.is_empty(),
            "At least one prediction should survive threshold=0.01"
        );
        let has_label_0 = heap2.iter().any(|&(_, idx)| idx == 0);
        assert!(has_label_0, "Label 0 should be in pruned predictions");
        for &(log_prob, label_idx) in &heap2 {
            assert!(
                label_idx >= 0 && label_idx < osz as i32,
                "Label index {} out of range [0, {})",
                label_idx,
                osz
            );
            assert!(log_prob.is_finite(), "Log-prob should be finite");
        }

        let mut heap3 = Predictions::new();
        loss.predict(4, 0.0, &mut heap3, &mut state);

        assert!(
            !heap3.is_empty(),
            "k=4 predict should return at least one result"
        );
        assert!(
            heap3.len() <= osz,
            "k=4 should return at most {} labels",
            osz
        );

        // Results should be sorted descending by log-prob
        for i in 1..heap3.len() {
            assert!(
                heap3[i - 1].0 >= heap3[i].0,
                "Results should be sorted descending: heap3[{}].0={} < heap3[{}].0={}",
                i - 1,
                heap3[i - 1].0,
                i,
                heap3[i].0
            );
        }
        // Label 0 should be first (highest score)
        assert_eq!(
            heap3[0].1, 0,
            "Label 0 should have highest score in k=4 prediction"
        );
    }

    // ── SoftmaxLoss ────────────────────────────────────────────────────────

    /// VAL-INF-004: Softmax output sums to 1.0 within 1e-5
    #[test]
    fn test_softmax_normalization() {
        let nlabels = 5usize;
        let dim = 4usize;
        let mut wo = DenseMatrix::new(nlabels as i64, dim as i64);
        for i in 0..nlabels {
            for j in 0..dim {
                *wo.at_mut(i as i64, j as i64) = ((i + j) as f32) * 0.1;
            }
        }
        let wo = Arc::new(wo);
        let loss = SoftmaxLoss::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, nlabels, 0);
        for i in 0..dim {
            state.hidden[i] = (i as f32 + 1.0) * 0.5;
        }

        loss.compute_output(&mut state);

        let sum: f32 = (0..nlabels).map(|i| state.output[i]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax output should sum to 1.0, got {}",
            sum
        );
        for i in 0..nlabels {
            assert!(
                state.output[i] >= 0.0 && state.output[i] <= 1.0,
                "Softmax output[{}]={} should be in [0,1]",
                i,
                state.output[i]
            );
        }
    }

    /// VAL-INF-004: Softmax numerical stability with extreme logits (no NaN/Inf)
    #[test]
    fn test_softmax_numerical_stability() {
        let nlabels = 3usize;
        let dim = 1usize;
        let mut wo = DenseMatrix::new(nlabels as i64, dim as i64);
        // hidden[0] = 1.0; wo[0]·h = 1000, wo[1]·h = -1000, wo[2]·h = 0
        *wo.at_mut(0, 0) = 1000.0;
        *wo.at_mut(1, 0) = -1000.0;
        *wo.at_mut(2, 0) = 0.0;
        let wo = Arc::new(wo);
        let loss = SoftmaxLoss::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, nlabels, 0);
        state.hidden[0] = 1.0;

        loss.compute_output(&mut state);

        for i in 0..nlabels {
            assert!(
                state.output[i].is_finite(),
                "Softmax output[{}]={} should be finite",
                i,
                state.output[i]
            );
        }
        let sum: f32 = (0..nlabels).map(|i| state.output[i]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax sum should be 1.0 with extreme logits, got {}",
            sum
        );
        assert!(
            state.output[0] > 0.99,
            "Label 0 should dominate with logit=1000, got {}",
            state.output[0]
        );
    }

    /// VAL-INF-004: SoftmaxLoss forward returns non-negative, finite value
    #[test]
    fn test_softmax_forward() {
        let nlabels = 3usize;
        let dim = 2usize;
        let wo = Arc::new(DenseMatrix::new(nlabels as i64, dim as i64));
        let loss = SoftmaxLoss::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, nlabels, 0);

        let targets = vec![1i32];
        let result = loss.forward(&targets, 0, &mut state, 0.1, false);
        assert!(
            result.is_finite(),
            "Softmax forward should be finite, got {}",
            result
        );
        assert!(result >= 0.0, "Softmax forward should be non-negative");
    }

    // ── OneVsAllLoss ───────────────────────────────────────────────────────

    /// VAL-INF-005: OVA forward returns finite, non-negative value
    #[test]
    fn test_ova_loss_forward() {
        let nlabels = 4usize;
        let dim = 3usize;
        let wo = Arc::new(DenseMatrix::new(nlabels as i64, dim as i64));
        let loss = OneVsAllLoss::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, nlabels, 0);
        state.hidden[0] = 1.0;
        state.hidden[1] = 0.5;

        let targets = vec![1i32]; // label 1 is positive
        let result = loss.forward(&targets, 0, &mut state, 0.1, false);
        assert!(
            result.is_finite(),
            "OVA forward should be finite, got {}",
            result
        );
        assert!(result >= 0.0, "OVA forward should be non-negative");
    }

    /// VAL-INF-005: OVA classes are independent (scores computed independently)
    #[test]
    fn test_ova_independence() {
        let nlabels = 3usize;
        let dim = 2usize;
        let mut wo = DenseMatrix::new(nlabels as i64, dim as i64);
        // wo[0] = [1, 0], wo[1] = [0, 1], wo[2] = [1, 1]
        *wo.at_mut(0, 0) = 1.0;
        *wo.at_mut(1, 1) = 1.0;
        *wo.at_mut(2, 0) = 1.0;
        *wo.at_mut(2, 1) = 1.0;
        let wo = Arc::new(wo);
        let loss = OneVsAllLoss::new(wo, Arc::new(LossTables::new()));

        let mut state = State::new(dim, nlabels, 0);
        state.hidden[0] = 2.0;
        state.hidden[1] = 2.0;

        loss.compute_output(&mut state);

        // Each output is sigmoid(wo[i] · hidden), independently
        let tables = LossTables::new();
        let score_0 = tables.sigmoid(2.0); // wo[0]·h = 1*2 + 0*2 = 2
        let score_1 = tables.sigmoid(2.0); // wo[1]·h = 0*2 + 1*2 = 2
        let score_2 = tables.sigmoid(4.0); // wo[2]·h = 1*2 + 1*2 = 4

        assert!(
            (state.output[0] - score_0).abs() < 1e-4,
            "OVA output[0] should be sigmoid(2), got {}",
            state.output[0]
        );
        assert!(
            (state.output[1] - score_1).abs() < 1e-4,
            "OVA output[1] should be sigmoid(2), got {}",
            state.output[1]
        );
        assert!(
            (state.output[2] - score_2).abs() < 1e-4,
            "OVA output[2] should be sigmoid(4), got {}",
            state.output[2]
        );

        // Sum of OVA outputs need NOT equal 1.0 (proving independence)
        let sum: f32 = (0..nlabels).map(|i| state.output[i]).sum();
        // sigmoid(2)≈0.88, sigmoid(2)≈0.88, sigmoid(4)≈0.98 → sum ≈ 2.74
        assert!(
            (sum - 1.0).abs() > 0.1,
            "OVA outputs should NOT sum to 1.0 (got {}), proving independence",
            sum
        );
    }

    /// VAL-INF-005: OVA gradient matches binary cross-entropy per class (independence property)
    ///
    /// Verifies that state.grad after forward+backprop equals the expected sum
    /// of per-class binary cross-entropy gradients:
    ///   state.grad = Σ_i  alpha_i * wo_original[i]
    /// where alpha_i = lr * (label_i - sigmoid(wo_original[i] · hidden))
    ///
    /// This proves gradient independence: class i's contribution depends only on
    /// wo[i] and is not affected by wo[j] for j≠i.
    #[test]
    fn test_ova_loss_gradient() {
        let nlabels = 3usize;
        let dim = 2usize;
        let mut wo = DenseMatrix::new(nlabels as i64, dim as i64);
        // wo[0] = [1, 0], wo[1] = [0, 1], wo[2] = [0.5, 0.5]
        *wo.at_mut(0, 0) = 1.0;
        *wo.at_mut(1, 1) = 1.0;
        *wo.at_mut(2, 0) = 0.5;
        *wo.at_mut(2, 1) = 0.5;
        let wo = Arc::new(wo);
        let loss = OneVsAllLoss::new(wo, Arc::new(LossTables::new()));

        let lr = 0.1f32;
        let mut state = State::new(dim, nlabels, 0);
        state.hidden[0] = 1.0;
        state.hidden[1] = 0.5;

        // class 1 is the positive target
        let targets = vec![1i32];
        loss.forward(&targets, 0, &mut state, lr, true);

        // Compute expected gradient using original wo values.
        // The binary_logistic method uses the current wo BEFORE it applies its own
        // Hogwild! update, so state.grad is computed with the original weights.
        let tables = LossTables::new();

        // class 0 (negative): dot = wo[0]·h = 1*1 + 0*0.5 = 1.0
        let alpha0 = lr * (0.0 - tables.sigmoid(1.0));
        // class 1 (positive): dot = wo[1]·h = 0*1 + 1*0.5 = 0.5
        let alpha1 = lr * (1.0 - tables.sigmoid(0.5));
        // class 2 (negative): dot = wo[2]·h = 0.5*1 + 0.5*0.5 = 0.75
        let alpha2 = lr * (0.0 - tables.sigmoid(0.75));

        // state.grad[0] = alpha0*wo[0][0] + alpha1*wo[1][0] + alpha2*wo[2][0]
        //               = alpha0*1 + alpha1*0 + alpha2*0.5
        let expected_g0 = alpha0 * 1.0 + alpha1 * 0.0 + alpha2 * 0.5;
        // state.grad[1] = alpha0*wo[0][1] + alpha1*wo[1][1] + alpha2*wo[2][1]
        //               = alpha0*0 + alpha1*1 + alpha2*0.5
        let expected_g1 = alpha0 * 0.0 + alpha1 * 1.0 + alpha2 * 0.5;

        assert!(
            (state.grad[0] - expected_g0).abs() < 1e-4,
            "OVA grad[0]: expected {:.6}, got {:.6}",
            expected_g0,
            state.grad[0]
        );
        assert!(
            (state.grad[1] - expected_g1).abs() < 1e-4,
            "OVA grad[1]: expected {:.6}, got {:.6}",
            expected_g1,
            state.grad[1]
        );

        // Verify independence: class 1's contribution to grad[1] is alpha1 * wo[1][1] = alpha1 * 1
        // class 2's contribution to grad[1] is alpha2 * wo[2][1] = alpha2 * 0.5
        // These are computed independently of wo[0]
        let class1_contrib_g1 = alpha1 * 1.0; // alpha1 * wo[1][1]
        let class2_contrib_g1 = alpha2 * 0.5; // alpha2 * wo[2][1]
        assert!(
            (state.grad[1] - (class1_contrib_g1 + class2_contrib_g1)).abs() < 1e-4,
            "Gradient independence: grad[1] should decompose as class1 ({:.6}) + class2 ({:.6}) = {:.6}, got {:.6}",
            class1_contrib_g1,
            class2_contrib_g1,
            class1_contrib_g1 + class2_contrib_g1,
            state.grad[1]
        );

        // All gradients must be finite
        for i in 0..dim {
            assert!(
                state.grad[i].is_finite(),
                "grad[{}] should be finite, got {}",
                i,
                state.grad[i]
            );
        }
    }

    // ── find_k_best ────────────────────────────────────────────────────────

    #[test]
    fn test_find_k_best_basic() {
        let mut output = Vector::new(5);
        output[0] = 0.1;
        output[1] = 0.5;
        output[2] = 0.3;
        output[3] = 0.05;
        output[4] = 0.8;

        let mut heap = Predictions::new();
        find_k_best(3, 0.0, &mut heap, &output);

        assert_eq!(heap.len(), 3);
        // Top-3 by log-prob: indices 4, 1, 2 (prob 0.8, 0.5, 0.3)
        assert_eq!(heap[0].1, 4); // highest prob
        assert_eq!(heap[1].1, 1);
        assert_eq!(heap[2].1, 2);
    }

    #[test]
    fn test_find_k_best_threshold() {
        let mut output = Vector::new(4);
        output[0] = 0.1;
        output[1] = 0.6;
        output[2] = 0.3;
        output[3] = 0.9;

        let mut heap = Predictions::new();
        find_k_best(10, 0.5, &mut heap, &output);

        // Only indices 1 (0.6) and 3 (0.9) are above threshold 0.5
        assert_eq!(heap.len(), 2);
        assert_eq!(heap[0].1, 3);
        assert_eq!(heap[1].1, 1);
    }

    #[test]
    fn test_find_k_best_k_zero() {
        let mut output = Vector::new(3);
        output[0] = 0.5;
        output[1] = 0.3;
        output[2] = 0.8;

        let mut heap = Predictions::new();
        find_k_best(0, 0.0, &mut heap, &output);
        assert_eq!(heap.len(), 0);
    }

    // ── std_log ────────────────────────────────────────────────────────────

    #[test]
    fn test_std_log_accuracy() {
        // std_log(x) = ln(x + 1e-5)
        let diff1 = (std_log(1.0) - (1.0f32 + 1e-5).ln()).abs();
        assert!(diff1 < 1e-6, "std_log(1.0) diff={}", diff1);
        let diff2 = (std_log(0.5) - (0.5f32 + 1e-5).ln()).abs();
        assert!(diff2 < 1e-6, "std_log(0.5) diff={}", diff2);
        // std_log is always finite for non-negative x
        assert!(std_log(0.0).is_finite());
    }

    // ── BinaryLogisticBase ─────────────────────────────────────────────────

    #[test]
    fn test_binary_logistic_positive_label() {
        let dim = 2usize;
        let wo = Arc::new(DenseMatrix::new(1, dim as i64));
        let base = BinaryLogisticBase::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, 1, 0);
        // All zeros: score = sigmoid(0) = 0.5
        // loss_positive = -log(0.5) > 0
        let loss = base.binary_logistic(0, &mut state, true, 0.0, false);
        assert!(
            loss > 0.0,
            "Positive label loss should be > 0 for zero weights"
        );
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_binary_logistic_negative_label() {
        let dim = 2usize;
        let wo = Arc::new(DenseMatrix::new(1, dim as i64));
        let base = BinaryLogisticBase::new(wo, Arc::new(LossTables::new()));
        let mut state = State::new(dim, 1, 0);
        // All zeros: score = sigmoid(0) = 0.5
        // loss_negative = -log(1 - 0.5) = -log(0.5) > 0
        let loss = base.binary_logistic(0, &mut state, false, 0.0, false);
        assert!(
            loss > 0.0,
            "Negative label loss should be > 0 for zero weights"
        );
        assert!(loss.is_finite(), "Loss should be finite");
    }

    // ── NegativeSamplingLoss::get_negative infinite-loop guard ─────────────

    /// Validates that `get_negative` terminates even when the entire negative
    /// sampling table is filled with a single label (degenerate table).
    ///
    /// With a 1-label corpus every entry in the negatives table equals 0.
    /// The old loop-forever implementation would spin indefinitely; the guarded
    /// version should return within MAX_RETRIES and produce a finite result.
    #[test]
    fn test_ns_get_negative_degenerate_single_label() {
        // Build a NegativeSamplingLoss with 2 labels but counts = [1_000_000, 0]
        // so that the negative sampling table is filled entirely with label 0.
        let wo = Arc::new(DenseMatrix::new(2, 4));
        // Give label 1 a count of 0; the table builder skips labels with
        // zero entries, so the table ends up all-zeros (all entries == 0).
        let counts = vec![1_000_000i64, 0];
        let loss = NegativeSamplingLoss::new(wo, 5, &counts, Arc::new(LossTables::new()));

        // All entries in negatives should be label 0.
        assert!(
            loss.negatives().iter().all(|&x| x == 0),
            "Expected all-zero negatives table for degenerate counts"
        );

        // get_negative(target=0, …) should NOT loop forever; it should return
        // the fallback value quickly (within MAX_RETRIES = 100 iterations).
        let mut rng = MinstdRng::new(42);
        // We call it many times to be sure it never blocks.
        for _ in 0..1000 {
            let neg = loss.get_negative(0, &mut rng);
            // In the degenerate single-label case the fallback is (0+1)%2 = 1.
            assert_eq!(
                neg, 1,
                "Fallback negative should be (target+1)%n_labels = 1"
            );
        }
    }

    /// Validates that `get_negative` returns a valid, different label in the
    /// normal (non-degenerate) case.
    #[test]
    fn test_ns_get_negative_returns_different_label() {
        let wo = Arc::new(DenseMatrix::new(4, 8));
        let counts = vec![100i64, 80, 60, 40];
        let loss = NegativeSamplingLoss::new(wo, 5, &counts, Arc::new(LossTables::new()));

        let mut rng = MinstdRng::new(123);
        // For target = 0, every returned negative should be != 0.
        for _ in 0..1000 {
            let neg = loss.get_negative(0, &mut rng);
            assert_ne!(
                neg, 0,
                "get_negative should return a label different from target (0)"
            );
        }
    }
}
