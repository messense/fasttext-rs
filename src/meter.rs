// Meter: precision, recall, F1, per-label metrics, precision-recall curves

use std::collections::HashMap;

/// Per-label statistics accumulated during evaluation.
#[derive(Debug, Default, Clone)]
struct LabelMetrics {
    /// Number of times this label appeared in the gold set.
    gold: u64,
    /// Number of times this label appeared in the model's predicted set.
    predicted: u64,
    /// Number of times this label was both predicted and in the gold set (true positives).
    predicted_gold: u64,
    /// Each entry is `(probability, gold_flag)` where `gold_flag = 1.0` if the example
    /// was truly positive for this label, else `0.0`.
    ///
    /// Used to compute the precision-recall curve.
    score_vs_true: Vec<(f32, f32)>,
}

impl LabelMetrics {
    /// Returns precision for this label, or 0.0 when no predictions were made.
    fn precision(&self) -> f64 {
        if self.predicted == 0 {
            0.0
        } else {
            self.predicted_gold as f64 / self.predicted as f64
        }
    }

    /// Returns recall for this label, or 0.0 when there are no gold examples.
    fn recall(&self) -> f64 {
        if self.gold == 0 {
            0.0
        } else {
            self.predicted_gold as f64 / self.gold as f64
        }
    }

    /// Returns F1 score for this label.
    ///
    /// Returns `0.0` (not NaN) when both precision and recall are zero.
    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Evaluation meter that accumulates precision, recall, F1, per-label metrics,
/// and precision-recall curve data.
///
/// Matches the behavior of C++ `fasttext::Meter`.
#[derive(Debug, Default)]
pub struct Meter {
    /// Total number of examples processed.
    n_examples: u64,
    /// Total gold labels across all examples.
    gold: u64,
    /// Total predicted labels across all examples (= k × n_examples for fixed k).
    predicted: u64,
    /// Total correctly predicted labels across all examples.
    predicted_gold: u64,
    /// Per-label statistics keyed by label ID.
    label_metrics: HashMap<i32, LabelMetrics>,
}

impl Meter {
    /// Create a new empty `Meter`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate one example's predictions and gold labels.
    ///
    /// # Arguments
    /// - `predictions`: slice of `(probability, label_id)` pairs returned by the model,
    ///   sorted by descending probability.
    /// - `gold_labels`: slice of true label IDs for this example.
    /// - `k`: the number of predictions that were *requested* (not the number actually
    ///   returned after threshold filtering).  This is used as the denominator for
    ///   precision, matching the C++ fastText convention:
    ///   `precision(k) = correct_in_top_k / (k * n_examples)`.
    pub fn add(&mut self, predictions: &[(f32, i32)], gold_labels: &[i32], k: usize) {
        self.n_examples += 1;
        self.gold += gold_labels.len() as u64;
        // Use k (not predictions.len()) so the denominator is always k * n_examples,
        // matching the C++ fastText precision(k) convention.
        self.predicted += k as u64;

        for &(prob, label_id) in predictions {
            let lm = self.label_metrics.entry(label_id).or_default();
            lm.predicted += 1;

            let is_gold = gold_labels.contains(&label_id);
            if is_gold {
                lm.predicted_gold += 1;
                self.predicted_gold += 1;
            }
            let gold_flag = if is_gold { 1.0f32 } else { 0.0f32 };
            lm.score_vs_true.push((prob, gold_flag));
        }

        // Accumulate gold counts for each true label.
        for &label_id in gold_labels {
            let lm = self.label_metrics.entry(label_id).or_default();
            lm.gold += 1;
        }
    }

    // -------------------------------------------------------------------------
    // Aggregate metrics
    // -------------------------------------------------------------------------

    /// Number of examples added to the meter.
    pub fn n_examples(&self) -> u64 {
        self.n_examples
    }

    /// Aggregate precision: `predictedGold / predicted`.
    ///
    /// Returns `0.0` when no predictions were made.
    pub fn precision(&self) -> f64 {
        if self.predicted == 0 {
            0.0
        } else {
            self.predicted_gold as f64 / self.predicted as f64
        }
    }

    /// Aggregate recall: `predictedGold / gold`.
    ///
    /// Returns `0.0` when there are no gold labels.
    pub fn recall(&self) -> f64 {
        if self.gold == 0 {
            0.0
        } else {
            self.predicted_gold as f64 / self.gold as f64
        }
    }

    /// Aggregate F1 score: harmonic mean of precision and recall.
    ///
    /// Returns `0.0` (not NaN) when both precision and recall are zero.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    // -------------------------------------------------------------------------
    // Per-label metrics
    // -------------------------------------------------------------------------

    /// Precision for a specific label.
    ///
    /// Returns `0.0` if the label was never predicted or is unknown.
    pub fn precision_for_label(&self, label_id: i32) -> f64 {
        self.label_metrics
            .get(&label_id)
            .map(|lm| lm.precision())
            .unwrap_or(0.0)
    }

    /// Recall for a specific label.
    ///
    /// Returns `0.0` if the label never appeared in the gold set or is unknown.
    pub fn recall_for_label(&self, label_id: i32) -> f64 {
        self.label_metrics
            .get(&label_id)
            .map(|lm| lm.recall())
            .unwrap_or(0.0)
    }

    /// F1 score for a specific label.
    ///
    /// Returns `0.0` (not NaN) when both precision and recall are zero, or when unknown.
    pub fn f1_for_label(&self, label_id: i32) -> f64 {
        self.label_metrics
            .get(&label_id)
            .map(|lm| lm.f1())
            .unwrap_or(0.0)
    }

    // -------------------------------------------------------------------------
    // Precision-recall curve
    // -------------------------------------------------------------------------

    /// Returns the precision-recall curve for a specific label.
    ///
    /// Each element is `(threshold, precision, recall)` where `threshold` is the
    /// probability score at which the decision boundary was evaluated.
    ///
    /// The curve is sorted by **decreasing threshold** (highest threshold first),
    /// so as you iterate from first to last, threshold decreases and recall
    /// non-decreases.
    ///
    /// Returns an empty vector if the label is unknown or has no gold examples.
    pub fn precision_recall_curve_for_label(&self, label_id: i32) -> Vec<(f64, f64, f64)> {
        let lm = match self.label_metrics.get(&label_id) {
            Some(lm) => lm,
            None => return Vec::new(),
        };

        let golds = lm.gold;
        if golds == 0 {
            return Vec::new();
        }

        // Sort (score, gold_flag) ascending by score (C++ `sort(ret.begin(), ret.end())`).
        let mut score_vs_true = lm.score_vs_true.clone();
        score_vs_true.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Build cumulative (TP, FP) counts iterating from highest score.
        // Matches C++ `getPositiveCounts`.
        let mut positive_counts: Vec<(u64, u64)> = Vec::new();
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut last_score = f32::INFINITY;

        for &(score, gold_flag) in score_vs_true.iter().rev() {
            if score < 0.0 {
                // Sentinel value for false negatives; stop here.
                break;
            }
            if gold_flag == 1.0 {
                tp += 1;
            } else {
                fp += 1;
            }
            // Squeeze tied scores: update the last entry rather than adding a new one.
            if (score - last_score).abs() < f32::EPSILON && !positive_counts.is_empty() {
                *positive_counts.last_mut().unwrap() = (tp, fp);
            } else {
                positive_counts.push((tp, fp));
            }
            last_score = score;
        }

        if positive_counts.is_empty() {
            return Vec::new();
        }

        // Find the point of full recall (first where TP >= golds) and advance one
        // past it, matching C++ `std::next(lower_bound(...))`.
        let full_recall_idx = positive_counts
            .iter()
            .position(|&(tp_count, _)| tp_count >= golds)
            .map(|idx| idx + 1)
            .unwrap_or(positive_counts.len());

        // Reconstruct the threshold (score) for each distinct step by re-iterating
        // score_vs_true in reverse and collecting unique scores.
        let mut thresholds: Vec<f64> = Vec::new();
        let mut last_th = f32::INFINITY;
        for &(score, _gold_flag) in score_vs_true.iter().rev() {
            if score < 0.0 {
                break;
            }
            if (score - last_th).abs() >= f32::EPSILON || last_th.is_infinite() {
                thresholds.push(score as f64);
                last_th = score;
            }
        }

        // Build the PR curve from the first entry up to `full_recall_idx` (exclusive).
        let mut curve: Vec<(f64, f64, f64)> = Vec::new();
        for (i, &(tp_count, fp_count)) in positive_counts[..full_recall_idx].iter().enumerate() {
            let precision = if tp_count + fp_count == 0 {
                0.0
            } else {
                tp_count as f64 / (tp_count + fp_count) as f64
            };
            let recall = tp_count as f64 / golds as f64;
            let threshold = thresholds.get(i).copied().unwrap_or(0.0);
            curve.push((threshold, precision, recall));
        }

        curve
    }

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------

    /// Write general metrics (N, P@k, R@k) to `stdout`.
    ///
    /// Mirrors `Meter::writeGeneralMetrics` from C++ fastText.
    pub fn write_general_metrics(&self, k: i32) {
        println!("N\t{}", self.n_examples);
        println!("P@{}\t{:.3}", k, self.precision());
        println!("R@{}\t{:.3}", k, self.recall());
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // VAL-TRAIN-009: test_meter_precision_recall_f1
    // -------------------------------------------------------------------------

    #[test]
    fn test_meter_precision_recall_f1() {
        // 3 examples, k=1 prediction each.
        // Example 1: gold=[0], pred=[(0.9, 0)]   → correct
        // Example 2: gold=[1], pred=[(0.7, 1)]   → correct
        // Example 3: gold=[0], pred=[(0.5, 1)]   → wrong (predicts 1 but gold is 0)

        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 1);
        meter.add(&[(0.7, 1)], &[1], 1);
        meter.add(&[(0.5, 1)], &[0], 1);

        assert_eq!(meter.n_examples(), 3);

        // predicted_gold = 2 (examples 1 and 2)
        // k=1, n_examples=3 → predicted = k * n_examples = 1 * 3 = 3
        // precision = 2/3, recall = 2/3
        let p = meter.precision();
        let r = meter.recall();
        let f = meter.f1();
        let expected_p = 2.0 / 3.0;
        let expected_r = 2.0 / 3.0;
        let expected_f = 2.0 * expected_p * expected_r / (expected_p + expected_r);

        assert!(
            (p - expected_p).abs() < 1e-9,
            "Expected precision {:.6}, got {:.6}",
            expected_p,
            p
        );
        assert!(
            (r - expected_r).abs() < 1e-9,
            "Expected recall {:.6}, got {:.6}",
            expected_r,
            r
        );
        assert!(
            (f - expected_f).abs() < 1e-9,
            "Expected F1 {:.6}, got {:.6}",
            expected_f,
            f
        );
    }

    #[test]
    fn test_meter_precision_k2() {
        // 2 examples, k=2 predictions each.
        // Example 1: gold=[0], preds=[(0.9, 0), (0.6, 1)]  → 1 correct out of 2 predicted
        // Example 2: gold=[1, 2], preds=[(0.8, 1), (0.7, 2)] → 2 correct out of 2 predicted
        // precision = 3 / 4 = 0.75
        // recall = 3 / 3 = 1.0

        let mut meter = Meter::new();
        meter.add(&[(0.9, 0), (0.6, 1)], &[0], 2);
        meter.add(&[(0.8, 1), (0.7, 2)], &[1, 2], 2);

        let p = meter.precision();
        let r = meter.recall();

        assert!(
            (p - 0.75).abs() < 1e-9,
            "Expected precision 0.75, got {:.6}",
            p
        );
        assert!(
            (r - 1.0).abs() < 1e-9,
            "Expected recall 1.0, got {:.6}",
            r
        );
    }

    #[test]
    fn test_meter_recall_formula() {
        // precision(k) = (correct in top-k) / (k * n_examples)
        // recall(k) = (correct in top-k) / (total true labels)
        // 3 examples, k=1. Gold labels total = 5 (some multi-label).
        // Example 1: gold=[0, 1], pred=[(0.9, 0)]  → 1 correct
        // Example 2: gold=[2], pred=[(0.8, 2)]     → 1 correct
        // Example 3: gold=[3, 4], pred=[(0.7, 0)]  → 0 correct

        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0, 1], 1);
        meter.add(&[(0.8, 2)], &[2], 1);
        meter.add(&[(0.7, 0)], &[3, 4], 1);

        // predicted_gold = 2, predicted = 3, gold = 5
        assert!(
            (meter.precision() - 2.0 / 3.0).abs() < 1e-9,
            "Precision should be 2/3"
        );
        assert!(
            (meter.recall() - 2.0 / 5.0).abs() < 1e-9,
            "Recall should be 2/5"
        );
    }

    // -------------------------------------------------------------------------
    // Zero division handling
    // -------------------------------------------------------------------------

    #[test]
    fn test_meter_zero_predictions() {
        // No predictions have been added.
        let meter = Meter::new();
        assert_eq!(meter.precision(), 0.0, "Empty meter precision should be 0.0");
        assert_eq!(meter.recall(), 0.0, "Empty meter recall should be 0.0");
        assert_eq!(meter.f1(), 0.0, "Empty meter F1 should be 0.0");
    }

    #[test]
    fn test_meter_f1_zero_when_both_zero() {
        // If all predictions are wrong, precision = 0 and recall = 0.
        // F1 must be 0.0, NOT NaN.
        let mut meter = Meter::new();
        // Predict label 1, but gold is label 0.
        meter.add(&[(0.9, 1)], &[0], 1);

        assert_eq!(
            meter.precision(),
            0.0,
            "Precision should be 0.0 when all predictions wrong"
        );
        assert_eq!(
            meter.recall(),
            0.0,
            "Recall should be 0.0 when no correct predictions"
        );
        let f = meter.f1();
        assert!(
            f.is_finite(),
            "F1 should be finite (not NaN/Inf), got {}",
            f
        );
        assert_eq!(f, 0.0, "F1 should be 0.0 when precision and recall are both 0");
    }

    #[test]
    fn test_meter_f1_not_nan_per_label() {
        // Per-label F1 should also be 0.0, not NaN, when no TP.
        let mut meter = Meter::new();
        // Predict label 0 (wrong), gold is label 1.
        meter.add(&[(0.9, 0)], &[1], 1);

        // For label 0: predicted=1, predictedGold=0, gold=0 → p=0.0, r=0.0
        let f0 = meter.f1_for_label(0);
        assert!(
            f0.is_finite(),
            "Per-label F1 should be finite, got {}",
            f0
        );
        assert_eq!(f0, 0.0);

        // For label 1: predicted=0, predictedGold=0, gold=1 → p=0.0, r=0.0
        let f1 = meter.f1_for_label(1);
        assert!(
            f1.is_finite(),
            "Per-label F1 should be finite, got {}",
            f1
        );
        assert_eq!(f1, 0.0);

        // Unknown label
        let f99 = meter.f1_for_label(99);
        assert_eq!(f99, 0.0);
    }

    #[test]
    fn test_meter_no_gold_labels() {
        // If there are no gold labels, recall should be 0.0.
        let mut meter = Meter::new();
        // Example with empty gold set.
        meter.add(&[(0.9, 0)], &[], 1);

        assert_eq!(meter.n_examples(), 1);
        assert_eq!(meter.gold, 0);
        assert_eq!(meter.recall(), 0.0);
    }

    // -------------------------------------------------------------------------
    // VAL-TRAIN-009: test_meter_per_label
    // -------------------------------------------------------------------------

    #[test]
    fn test_meter_per_label() {
        // Build a scenario with known per-label TP/FP/FN:
        // Label 0: TP=2, FP=1, FN=0  → precision=2/3, recall=2/2=1.0
        // Label 1: TP=1, FP=0, FN=1  → precision=1/1=1.0, recall=1/2=0.5

        let mut meter = Meter::new();

        // Example 1: gold=[0], preds=[(0.9, 0)]     → label 0 TP
        meter.add(&[(0.9, 0)], &[0], 1);
        // Example 2: gold=[0, 1], preds=[(0.8, 0), (0.7, 1)]  → label 0 TP, label 1 TP
        meter.add(&[(0.8, 0), (0.7, 1)], &[0, 1], 2);
        // Example 3: gold=[1], preds=[(0.6, 0)]     → label 0 FP, label 1 FN (not predicted)
        meter.add(&[(0.6, 0)], &[1], 1);

        // Label 0: gold=2, predicted=3, predicted_gold=2
        let p0 = meter.precision_for_label(0);
        let r0 = meter.recall_for_label(0);
        let f0 = meter.f1_for_label(0);

        assert!(
            (p0 - 2.0 / 3.0).abs() < 1e-9,
            "Label 0 precision should be 2/3, got {:.6}",
            p0
        );
        assert!(
            (r0 - 1.0).abs() < 1e-9,
            "Label 0 recall should be 1.0, got {:.6}",
            r0
        );
        let expected_f0 = 2.0 * (2.0 / 3.0) * 1.0 / (2.0 / 3.0 + 1.0);
        assert!(
            (f0 - expected_f0).abs() < 1e-9,
            "Label 0 F1 should be {:.6}, got {:.6}",
            expected_f0,
            f0
        );

        // Label 1: gold=2, predicted=1, predicted_gold=1
        let p1 = meter.precision_for_label(1);
        let r1 = meter.recall_for_label(1);
        let f1 = meter.f1_for_label(1);

        assert!(
            (p1 - 1.0).abs() < 1e-9,
            "Label 1 precision should be 1.0, got {:.6}",
            p1
        );
        assert!(
            (r1 - 0.5).abs() < 1e-9,
            "Label 1 recall should be 0.5, got {:.6}",
            r1
        );
        let expected_f1 = 2.0 * 1.0 * 0.5 / (1.0 + 0.5);
        assert!(
            (f1 - expected_f1).abs() < 1e-9,
            "Label 1 F1 should be {:.6}, got {:.6}",
            expected_f1,
            f1
        );
    }

    #[test]
    fn test_meter_per_label_unknown() {
        let meter = Meter::new();
        assert_eq!(meter.precision_for_label(42), 0.0);
        assert_eq!(meter.recall_for_label(42), 0.0);
        assert_eq!(meter.f1_for_label(42), 0.0);
    }

    // -------------------------------------------------------------------------
    // VAL-TRAIN-009: test_meter_pr_curve
    // -------------------------------------------------------------------------

    #[test]
    fn test_meter_pr_curve() {
        // Construct a scenario with a known PR curve for label 0.
        // All predictions for label 0 are correct (perfect classifier).
        // Example 1: gold=[0], pred=[(0.9, 0)]
        // Example 2: gold=[0], pred=[(0.7, 0)]
        // Example 3: gold=[1], pred=[(0.5, 1)]  (different label)

        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 1);
        meter.add(&[(0.7, 0)], &[0], 1);
        meter.add(&[(0.5, 1)], &[1], 1);

        let curve = meter.precision_recall_curve_for_label(0);

        // Should have 2 points (one per distinct threshold).
        assert_eq!(curve.len(), 2, "Expected 2 curve points, got {}", curve.len());

        // All precision and recall values must be in [0, 1].
        for &(threshold, precision, recall) in &curve {
            assert!(
                (0.0..=1.0).contains(&precision),
                "Precision {:.4} out of [0,1] at threshold {:.4}",
                precision,
                threshold
            );
            assert!(
                (0.0..=1.0).contains(&recall),
                "Recall {:.4} out of [0,1] at threshold {:.4}",
                recall,
                threshold
            );
        }

        // First point: threshold=0.9, precision=1.0, recall=0.5
        let (t0, p0, r0) = curve[0];
        assert!(
            (t0 - 0.9).abs() < 1e-4,
            "First threshold should be ~0.9, got {:.4}",
            t0
        );
        assert!(
            (p0 - 1.0).abs() < 1e-9,
            "First precision should be 1.0, got {:.6}",
            p0
        );
        assert!(
            (r0 - 0.5).abs() < 1e-9,
            "First recall should be 0.5, got {:.6}",
            r0
        );

        // Second point: threshold=0.7, precision=1.0, recall=1.0
        let (t1, p1, r1) = curve[1];
        assert!(
            (t1 - 0.7).abs() < 1e-4,
            "Second threshold should be ~0.7, got {:.4}",
            t1
        );
        assert!(
            (p1 - 1.0).abs() < 1e-9,
            "Second precision should be 1.0, got {:.6}",
            p1
        );
        assert!(
            (r1 - 1.0).abs() < 1e-9,
            "Second recall should be 1.0, got {:.6}",
            r1
        );

        // Verify monotonicity: as threshold decreases (from first to last),
        // recall should be non-decreasing.
        for i in 1..curve.len() {
            let (_, _, r_prev) = curve[i - 1];
            let (_, _, r_curr) = curve[i];
            assert!(
                r_curr >= r_prev - 1e-9,
                "Recall should be non-decreasing as threshold decreases: \
                 curve[{}].recall={:.4} < curve[{}].recall={:.4}",
                i,
                r_curr,
                i - 1,
                r_prev
            );
        }

        // Verify thresholds are sorted in decreasing order.
        for i in 1..curve.len() {
            let (t_prev, _, _) = curve[i - 1];
            let (t_curr, _, _) = curve[i];
            assert!(
                t_curr <= t_prev + 1e-9,
                "Thresholds should be non-increasing: \
                 curve[{}].threshold={:.4} > curve[{}].threshold={:.4}",
                i,
                t_curr,
                i - 1,
                t_prev
            );
        }
    }

    #[test]
    fn test_meter_pr_curve_with_fp() {
        // Scenario with some false positives for label 0:
        // Example 1: gold=[0], pred=[(0.9, 0)]   → TP@0.9 for label 0
        // Example 2: gold=[1], pred=[(0.7, 0)]   → FP@0.7 for label 0
        // Example 3: gold=[0], pred=[(0.5, 0)]   → TP@0.5 for label 0

        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 1);
        meter.add(&[(0.7, 0)], &[1], 1);
        meter.add(&[(0.5, 0)], &[0], 1);

        let curve = meter.precision_recall_curve_for_label(0);

        // Must have some points.
        assert!(!curve.is_empty(), "PR curve should be non-empty");

        // All values must be in [0, 1].
        for &(threshold, precision, recall) in &curve {
            assert!(
                (0.0..=1.0).contains(&precision),
                "Precision {:.4} out of range at threshold {:.4}",
                precision,
                threshold
            );
            assert!(
                (0.0..=1.0).contains(&recall),
                "Recall {:.4} out of range at threshold {:.4}",
                recall,
                threshold
            );
            assert!(
                threshold >= 0.0,
                "Threshold should be non-negative, got {:.4}",
                threshold
            );
        }

        // Thresholds are sorted in decreasing order.
        for i in 1..curve.len() {
            let (t_prev, _, _) = curve[i - 1];
            let (t_curr, _, _) = curve[i];
            assert!(
                t_curr <= t_prev + 1e-9,
                "Thresholds should be non-increasing"
            );
        }

        // Recall should be non-decreasing as threshold decreases.
        for i in 1..curve.len() {
            let (_, _, r_prev) = curve[i - 1];
            let (_, _, r_curr) = curve[i];
            assert!(
                r_curr >= r_prev - 1e-9,
                "Recall should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_meter_pr_curve_unknown_label() {
        let meter = Meter::new();
        let curve = meter.precision_recall_curve_for_label(99);
        assert!(
            curve.is_empty(),
            "PR curve for unknown label should be empty"
        );
    }

    #[test]
    fn test_meter_pr_curve_no_gold() {
        // Label predicted but never in gold set.
        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[1], 1); // predict 0, gold is 1

        let curve = meter.precision_recall_curve_for_label(0);
        assert!(
            curve.is_empty(),
            "PR curve should be empty when label never in gold set"
        );
    }

    // -------------------------------------------------------------------------
    // Additional correctness tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meter_n_examples() {
        let mut meter = Meter::new();
        assert_eq!(meter.n_examples(), 0);
        meter.add(&[], &[], 0);
        assert_eq!(meter.n_examples(), 1);
        meter.add(&[(0.5, 0)], &[0], 1);
        assert_eq!(meter.n_examples(), 2);
    }

    #[test]
    fn test_meter_all_correct() {
        // 4 examples, all correct.
        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 1);
        meter.add(&[(0.8, 1)], &[1], 1);
        meter.add(&[(0.7, 2)], &[2], 1);
        meter.add(&[(0.6, 0)], &[0], 1);

        assert!(
            (meter.precision() - 1.0).abs() < 1e-9,
            "All correct, precision should be 1.0"
        );
        assert!(
            (meter.recall() - 1.0).abs() < 1e-9,
            "All correct, recall should be 1.0"
        );
        assert!(
            (meter.f1() - 1.0).abs() < 1e-9,
            "All correct, F1 should be 1.0"
        );
    }

    #[test]
    fn test_meter_all_wrong() {
        // All predictions wrong.
        let mut meter = Meter::new();
        meter.add(&[(0.9, 1)], &[0], 1);
        meter.add(&[(0.8, 0)], &[1], 1);

        assert_eq!(meter.precision(), 0.0, "All wrong, precision should be 0.0");
        assert_eq!(meter.recall(), 0.0, "All wrong, recall should be 0.0");
        assert_eq!(meter.f1(), 0.0, "All wrong, F1 should be 0.0 (not NaN)");
    }

    #[test]
    fn test_meter_write_general_metrics() {
        // Smoke test that write_general_metrics doesn't panic.
        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 1);
        meter.add(&[(0.7, 1)], &[1], 1);
        // Just verify it runs without panicking.
        meter.write_general_metrics(1);
    }

    // -------------------------------------------------------------------------
    // Precision(k) denominator: k * n_examples (matching fastText convention)
    // -------------------------------------------------------------------------

    /// Verify precision(k) uses k * n_examples as denominator, not actual predictions.
    ///
    /// When threshold filtering reduces the number of returned predictions below k,
    /// the denominator must still be k * n_examples, not the actual prediction count.
    /// This matches the C++ fastText convention:
    ///   precision(k) = correct_in_top_k / (k * n_examples)
    #[test]
    fn test_meter_precision_k_denominator_threshold_filtering() {
        // Simulate: k=5 requested, but only 1 prediction returned per example
        // (as if threshold filtering reduced predictions below k).
        // Example 1: correct (pred label 0, gold label 0)
        // Example 2: wrong   (pred label 1, gold label 0)
        let mut meter = Meter::new();
        meter.add(&[(0.9, 0)], &[0], 5);  // k=5 requested, 1 returned
        meter.add(&[(0.8, 1)], &[0], 5);  // k=5 requested, 1 returned

        assert_eq!(meter.n_examples(), 2);

        // predicted_gold = 1 (only example 1 is correct)
        // denominator = k * n_examples = 5 * 2 = 10
        let p = meter.precision();
        assert!(
            (p - 0.1).abs() < 1e-9,
            "Precision should be 1/(k*n) = 1/(5*2) = 0.1 (not 1/2 = 0.5), got {:.6}",
            p
        );

        // recall is unaffected: predicted_gold / gold = 1 / 2
        let r = meter.recall();
        assert!(
            (r - 0.5).abs() < 1e-9,
            "Recall should be 1/2 = 0.5, got {:.6}",
            r
        );
    }

    /// Verify that when all k predictions are returned (no threshold filtering),
    /// precision = predicted_gold / (k * n_examples) still matches the old formula
    /// predicted_gold / actual_prediction_count.
    #[test]
    fn test_meter_precision_k_full_predictions() {
        // k=2, all 2 predictions always returned: denominator = 2*2 = 4
        // Example 1: gold=[0], preds=[(0.9,0),(0.7,1)] → 1 correct
        // Example 2: gold=[1,2], preds=[(0.8,1),(0.6,2)] → 2 correct
        let mut meter = Meter::new();
        meter.add(&[(0.9, 0), (0.7, 1)], &[0], 2);
        meter.add(&[(0.8, 1), (0.6, 2)], &[1, 2], 2);

        // predicted_gold = 3, denominator = k*n = 2*2 = 4
        let p = meter.precision();
        assert!(
            (p - 0.75).abs() < 1e-9,
            "Precision should be 3/4 = 0.75, got {:.6}",
            p
        );
    }
}
