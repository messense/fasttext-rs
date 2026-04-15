// ProductQuantizer: k-means product quantization (256 centroids, 8-bit codes)
//
// Matches the C++ fastText `ProductQuantizer` class with identical binary format
// and computation semantics.

use std::io::{Read, Write};

use crate::error::{FastTextError, Result};
use crate::model::MinstdRng;
use crate::utils;
use crate::vector::Vector;

// Constants

/// Number of bits per code. Each sub-quantizer assigns a code in [0, 2^NBITS).
pub const NBITS: i32 = 8;

/// Number of centroids per sub-quantizer: 2^NBITS = 256.
pub const KSUB: i32 = 1 << NBITS;

/// Maximum number of data points used in k-means training (256 * 256 = 65536).
pub const MAX_POINTS: i32 = 256 * KSUB;

/// Number of k-means EM iterations per sub-quantizer.
pub const NITER: i32 = 25;

/// Seed for the internal RNG (matches C++ `seed_ = 1234`).
const SEED: u64 = 1234;

/// Small perturbation applied when splitting empty clusters (matches C++ `eps_ = 1e-7`).
const EPS: f32 = 1e-7;
// Helper: L2 squared distance

/// Compute the squared Euclidean distance between two equal-length f32 slices.
#[inline]
fn dist_l2(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

// E-step and M-step helpers (free functions to avoid borrow conflicts)

/// Assign the vector `x` (length `d`) to the nearest of the `ksub` centroids
/// laid out contiguously in `centroids` (length `ksub * d`).
///
/// Writes the chosen centroid index into `code` and returns the squared distance.
fn assign_centroid(x: &[f32], centroids: &[f32], code: &mut u8, d: usize) -> f32 {
    let first = &centroids[..d];
    let mut best_dist = dist_l2(x, first);
    *code = 0;
    for j in 1..KSUB as usize {
        let c = &centroids[j * d..(j + 1) * d];
        let dist = dist_l2(x, c);
        if dist < best_dist {
            *code = j as u8;
            best_dist = dist;
        }
    }
    best_dist
}

/// E-step: assign each of the `n` data points in `x` (shape `n × d`) to its
/// nearest centroid in `centroids` (shape `KSUB × d`).
fn estep(x: &[f32], centroids: &[f32], codes: &mut [u8], d: usize, n: usize) {
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        assign_centroid(xi, centroids, &mut codes[i], d);
    }
}

/// M-step: recompute `centroids` (shape `KSUB × d`) as the mean of assigned
/// points. Empty clusters are filled by splitting the largest cluster with a
/// small perturbation, using `rng` for random tie-breaking.
fn mstep(rng: &mut MinstdRng, x: &[f32], centroids: &mut [f32], codes: &[u8], d: usize, n: usize) {
    let ksub = KSUB as usize;
    let mut nelts = vec![0i32; ksub];

    // Zero centroids accumulator.
    for v in centroids[..d * ksub].iter_mut() {
        *v = 0.0;
    }

    // Accumulate point sums.
    for i in 0..n {
        let k = codes[i] as usize;
        let xi = &x[i * d..(i + 1) * d];
        let c = &mut centroids[k * d..(k + 1) * d];
        for (cj, &xj) in c.iter_mut().zip(xi.iter()) {
            *cj += xj;
        }
        nelts[k] += 1;
    }

    // Normalize by cluster size.
    for k in 0..ksub {
        let z = nelts[k] as f32;
        if z != 0.0 {
            let c = &mut centroids[k * d..(k + 1) * d];
            for cj in c.iter_mut() {
                *cj /= z;
            }
        }
    }

    // Handle empty clusters by splitting a non-trivial cluster.
    for k in 0..ksub {
        if nelts[k] == 0 {
            let mut m = 0usize;
            // Find a cluster with more than 1 point to split.
            while rng.uniform_real() * (n as f64 - ksub as f64) >= (nelts[m] - 1) as f64 {
                m = (m + 1) % ksub;
            }
            // Copy centroid m to centroid k, then perturb both.
            let (k_start, m_start) = (k * d, m * d);
            for j in 0..d {
                centroids[k_start + j] = centroids[m_start + j];
            }
            for j in 0..d {
                let sign = if j % 2 == 1 { 1.0f32 } else { -1.0f32 };
                centroids[k_start + j] += sign * EPS;
                centroids[m_start + j] -= sign * EPS;
            }
            nelts[k] = nelts[m] / 2;
            nelts[m] -= nelts[k];
        }
    }
}

/// Run `niter` iterations of k-means on `n` points in `d` dimensions, writing
/// `KSUB` centroids into `centroids` (length `KSUB * d`).
fn kmeans(rng: &mut MinstdRng, x: &[f32], centroids: &mut [f32], n: usize, d: usize) {
    let ksub = KSUB as usize;

    // Initialise centroids from a random permutation of the data points.
    let mut perm: Vec<i32> = (0..n as i32).collect();
    rng.shuffle(&mut perm);
    for i in 0..ksub {
        let row = perm[i] as usize;
        centroids[i * d..(i + 1) * d].copy_from_slice(&x[row * d..row * d + d]);
    }

    // EM iterations.
    let mut codes = vec![0u8; n];
    for _ in 0..NITER {
        estep(x, centroids, &mut codes, d, n);
        mstep(rng, x, centroids, &codes, d, n);
    }
}

// ProductQuantizer

/// Product Quantizer for approximate inner-product computation.
///
/// Splits a `dim`-dimensional vector into `nsubq` sub-vectors and quantizes
/// each independently with `KSUB = 256` centroids. Each code is a single `u8`.
///
/// Matches the C++ fastText `ProductQuantizer` class exactly in binary format
/// and computation semantics.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Original embedding dimension.
    pub dim: i32,
    /// Number of sub-quantizers.
    pub nsubq: i32,
    /// Sub-dimension for all but the last sub-quantizer.
    pub dsub: i32,
    /// Sub-dimension for the last sub-quantizer
    /// (equals `dsub` when `dim` is evenly divisible by `dsub`).
    pub lastdsub: i32,
    /// Centroid storage: `dim * KSUB` f32 values in layout described below.
    ///
    /// For sub-quantizer `m` and centroid `i`:
    /// - m < nsubq-1: starts at `(m * KSUB + i) * dsub`
    /// - m == nsubq-1: starts at `m * KSUB * dsub + i * lastdsub`
    pub centroids: Vec<f32>,
    /// Internal RNG for training.
    rng: MinstdRng,
}

impl ProductQuantizer {
    /// Create a new `ProductQuantizer` for vectors of length `dim`, splitting
    /// each into sub-vectors of length `dsub`.
    ///
    /// Matches C++ `ProductQuantizer(int32_t dim, int32_t dsub)`.
    pub fn new(dim: i32, dsub: i32) -> Self {
        let nsubq_base = dim / dsub;
        let rem = dim % dsub;
        let (nsubq, lastdsub) = if rem == 0 {
            (nsubq_base, dsub)
        } else {
            (nsubq_base + 1, rem)
        };
        ProductQuantizer {
            dim,
            nsubq,
            dsub,
            lastdsub,
            centroids: vec![0.0f32; (dim * KSUB) as usize],
            rng: MinstdRng::new(SEED),
        }
    }

    /// Return the byte offset and element count of centroid `i` for
    /// sub-quantizer `m` within `self.centroids`.
    #[inline]
    fn centroid_range(&self, m: usize, i: u8) -> (usize, usize) {
        let ksub = KSUB as usize;
        let dsub = self.dsub as usize;
        let nsubq = self.nsubq as usize;
        let d = if m == nsubq - 1 {
            self.lastdsub as usize
        } else {
            dsub
        };
        let offset = m * ksub * dsub + i as usize * d;
        (offset, d)
    }

    /// Borrow the centroid slice for sub-quantizer `m`, centroid index `i`.
    #[inline]
    pub fn get_centroids(&self, m: usize, i: u8) -> &[f32] {
        let (off, d) = self.centroid_range(m, i);
        &self.centroids[off..off + d]
    }

    /// Mutably borrow the centroid slice for sub-quantizer `m`, centroid `i`.
    #[inline]
    pub fn get_centroids_mut(&mut self, m: usize, i: u8) -> &mut [f32] {
        let (off, d) = self.centroid_range(m, i);
        &mut self.centroids[off..off + d]
    }

    /// Train the quantizer on `n` rows of the matrix `x` (row-major, `n × dim`).
    ///
    /// For each sub-quantizer, samples up to `MAX_POINTS` rows, then runs
    /// k-means for `NITER` iterations.
    ///
    /// When `n < KSUB` (fewer data points than centroids), the available data
    /// points are used directly as centroids, cycling through them to fill all
    /// `KSUB` slots.  This ensures `compute_code` always returns a valid code
    /// even for small datasets.
    ///
    /// Matches C++ `ProductQuantizer::train(int32_t n, const real* x)`.
    pub fn train(&mut self, n: i32, x: &[f32]) {
        if n <= 0 {
            return;
        }
        let n_usize = n as usize;
        let ksub = KSUB as usize;
        let dim = self.dim as usize;
        let dsub = self.dsub as usize;
        let nsubq = self.nsubq as usize;

        if n_usize < ksub {
            // Fewer data points than centroids.
            // Use the available points as centroids, cycling through them to
            // fill all KSUB slots so that compute_code works correctly.
            // The tie-breaking in assign_centroid (strict `<`) means duplicate
            // centroid slots (index >= n) are never preferred over their
            // originals (index < n), so every query is assigned to a code in
            // [0, n).
            for m in 0..nsubq {
                let d = if m == nsubq - 1 {
                    self.lastdsub as usize
                } else {
                    dsub
                };
                let cstart = m * ksub * dsub;
                for j in 0..ksub {
                    let row = j % n_usize;
                    let src = &x[row * dim + m * dsub..row * dim + m * dsub + d];
                    self.centroids[cstart + j * d..cstart + (j + 1) * d]
                        .copy_from_slice(src);
                }
            }
            return;
        }

        // n >= KSUB: normal k-means training.
        let np = n_usize.min(MAX_POINTS as usize);
        let mut perm: Vec<i32> = (0..n_usize as i32).collect();
        // Allocate max possible slice size (dsub or lastdsub).
        let max_d = dsub.max(self.lastdsub as usize);
        let mut xslice = vec![0.0f32; np * max_d];

        for m in 0..nsubq {
            let d = if m == nsubq - 1 {
                self.lastdsub as usize
            } else {
                dsub
            };

            // Reshuffle when subsampling.
            if np != n_usize {
                self.rng.shuffle(&mut perm);
            }

            // Extract sub-vectors for this sub-quantizer.
            for j in 0..np {
                let row = perm[j] as usize;
                let src = &x[row * dim + m * dsub..row * dim + m * dsub + d];
                xslice[j * d..(j + 1) * d].copy_from_slice(src);
            }

            // Centroid block for sub-quantizer m.
            let cstart = m * ksub * dsub;
            let clen = ksub * d;

            // Run k-means (borrow rng and centroids separately).
            let rng = &mut self.rng;
            kmeans(
                rng,
                &xslice[..np * d],
                &mut self.centroids[cstart..cstart + clen],
                np,
                d,
            );
        }
    }

    /// Assign the `dim`-dimensional vector `x` to its nearest centroid in each
    /// sub-quantizer. Writes `nsubq` bytes into `code`.
    ///
    /// Matches C++ `ProductQuantizer::compute_code(const real* x, uint8_t* code)`.
    pub fn compute_code(&self, x: &[f32], code: &mut [u8]) {
        let nsubq = self.nsubq as usize;
        let dsub = self.dsub as usize;
        let ksub = KSUB as usize;

        for m in 0..nsubq {
            let d = if m == nsubq - 1 {
                self.lastdsub as usize
            } else {
                dsub
            };
            let xi = &x[m * dsub..m * dsub + d];
            // All KSUB centroids for sub-quantizer m are contiguous starting at cstart.
            let cstart = m * ksub * dsub;
            assign_centroid(
                xi,
                &self.centroids[cstart..cstart + ksub * d],
                &mut code[m],
                d,
            );
        }
    }

    /// Encode all `n` rows of the matrix `x` (row-major, `n × dim`).
    /// Writes `n * nsubq` bytes into `codes`.
    ///
    /// Matches C++ `ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n)`.
    pub fn compute_codes(&self, x: &[f32], codes: &mut [u8], n: i32) {
        let n = n as usize;
        let dim = self.dim as usize;
        let nsubq = self.nsubq as usize;
        for i in 0..n {
            self.compute_code(
                &x[i * dim..(i + 1) * dim],
                &mut codes[i * nsubq..(i + 1) * nsubq],
            );
        }
    }

    /// Compute the approximate dot product `alpha * <x, reconstructed_row_t>`,
    /// where `reconstructed_row_t` is the concatenation of the centroids indexed
    /// by `codes[nsubq * t .. nsubq * (t+1)]`.
    ///
    /// Matches C++ `ProductQuantizer::mulcode(const Vector& x, const uint8_t* codes, int32_t t, real alpha)`.
    pub fn mulcode(&self, x: &Vector, codes: &[u8], t: i32, alpha: f32) -> f32 {
        let nsubq = self.nsubq as usize;
        let dsub = self.dsub as usize;
        let code = &codes[nsubq * t as usize..nsubq * (t as usize + 1)];

        let mut res = 0.0f32;
        for m in 0..nsubq {
            let d = if m == nsubq - 1 {
                self.lastdsub as usize
            } else {
                dsub
            };
            let c = self.get_centroids(m, code[m]);
            for n in 0..d {
                res += x[m * dsub + n] * c[n];
            }
        }
        res * alpha
    }

    /// Add `alpha * reconstructed_row_t` to the vector `x`, where
    /// `reconstructed_row_t` is the concatenation of the centroids indexed by
    /// `codes[nsubq * t .. nsubq * (t+1)]`.
    ///
    /// Matches C++ `ProductQuantizer::addcode(Vector& x, const uint8_t* codes, int32_t t, real alpha)`.
    pub fn addcode(&self, x: &mut Vector, codes: &[u8], t: i32, alpha: f32) {
        let nsubq = self.nsubq as usize;
        let dsub = self.dsub as usize;
        let code = &codes[nsubq * t as usize..nsubq * (t as usize + 1)];

        for m in 0..nsubq {
            let d = if m == nsubq - 1 {
                self.lastdsub as usize
            } else {
                dsub
            };
            let c = self.get_centroids(m, code[m]);
            for n in 0..d {
                x[m * dsub + n] += alpha * c[n];
            }
        }
    }

    /// Save the quantizer to `writer` in the C++ binary format.
    ///
    /// Format: `dim(i32)`, `nsubq(i32)`, `dsub(i32)`, `lastdsub(i32)`,
    /// then `dim * KSUB` f32 centroids.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        utils::write_i32(writer, self.dim)?;
        utils::write_i32(writer, self.nsubq)?;
        utils::write_i32(writer, self.dsub)?;
        utils::write_i32(writer, self.lastdsub)?;
        for &v in &self.centroids {
            utils::write_f32(writer, v)?;
        }
        Ok(())
    }

    /// Load a quantizer from `reader`.
    ///
    /// Reads the same binary format as `save`.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let dim = utils::read_i32(reader)?;
        let nsubq = utils::read_i32(reader)?;
        let dsub = utils::read_i32(reader)?;
        let lastdsub = utils::read_i32(reader)?;

        if dim < 0 || nsubq < 0 || dsub < 0 || lastdsub < 0 {
            return Err(FastTextError::InvalidModel(
                "ProductQuantizer: negative dimension field".to_string(),
            ));
        }

        let centroids_len = (dim * KSUB) as usize;
        let mut centroids = vec![0.0f32; centroids_len];
        for v in centroids.iter_mut() {
            *v = utils::read_f32(reader)?;
        }

        Ok(ProductQuantizer {
            dim,
            nsubq,
            dsub,
            lastdsub,
            centroids,
            rng: MinstdRng::new(SEED),
        })
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Constants

    #[test]
    fn test_pq_constants() {
        // VAL-DICT-015: PQ constants must match C++
        assert_eq!(NBITS, 8);
        assert_eq!(KSUB, 256);
        assert_eq!(MAX_POINTS, 65536);
        assert_eq!(NITER, 25);
    }

    // Dimension decomposition

    #[test]
    fn test_pq_dimension_decomposition_even() {
        // dim evenly divisible by dsub → lastdsub == dsub, nsubq == dim/dsub
        let pq = ProductQuantizer::new(100, 10);
        assert_eq!(pq.dim, 100);
        assert_eq!(pq.dsub, 10);
        assert_eq!(pq.nsubq, 10);
        assert_eq!(pq.lastdsub, 10); // lastdsub == dsub when evenly divisible
        assert_eq!(pq.centroids.len(), (100 * 256) as usize);
    }

    #[test]
    fn test_pq_dimension_decomposition_odd() {
        // dim=11, dsub=5 → 11/5=2 rem 1 → nsubq=3, lastdsub=1
        let pq = ProductQuantizer::new(11, 5);
        assert_eq!(pq.dim, 11);
        assert_eq!(pq.dsub, 5);
        assert_eq!(pq.nsubq, 3);
        assert_eq!(pq.lastdsub, 1);
        assert_eq!(pq.centroids.len(), (11 * 256) as usize);
    }

    #[test]
    fn test_pq_dimension_decomposition_dim_equals_dsub() {
        // dim == dsub → nsubq=1, lastdsub=dsub
        let pq = ProductQuantizer::new(8, 8);
        assert_eq!(pq.nsubq, 1);
        assert_eq!(pq.lastdsub, 8);
        assert_eq!(pq.centroids.len(), (8 * 256) as usize);
    }

    #[test]
    fn test_pq_dimension_decomposition_dsub_one() {
        // dsub=1 → nsubq=dim, lastdsub=1
        let pq = ProductQuantizer::new(4, 1);
        assert_eq!(pq.nsubq, 4);
        assert_eq!(pq.lastdsub, 1);
    }

    // compute_code: assign to nearest centroid

    /// Build a tiny PQ (dim=4, dsub=2, nsubq=2) with known centroids and
    /// return it. Used by several tests.
    ///
    /// Layout (2 sub-quantizers, each with 256 two-dimensional centroids):
    ///   sub-q 0, centroid 0: [1.0, 0.0]
    ///   sub-q 0, centroid 1: [0.0, 1.0]
    ///   all other centroids for sub-q 0: [100.0, 100.0]
    ///   sub-q 1, centroid 0: [2.0, 0.0]
    ///   sub-q 1, centroid 1: [0.0, 2.0]
    ///   all other centroids for sub-q 1: [100.0, 100.0]
    fn make_known_pq() -> ProductQuantizer {
        let mut pq = ProductQuantizer::new(4, 2);
        let ksub = KSUB as usize;
        let dsub = 2usize;

        // Fill all centroids with a far-away default.
        for v in pq.centroids.iter_mut() {
            *v = 100.0;
        }

        // Sub-quantizer 0 (m=0): cstart = 0*256*2 = 0
        let cstart0 = 0;
        pq.centroids[cstart0 + 0 * dsub] = 1.0; // centroid 0, dim 0
        pq.centroids[cstart0 + 0 * dsub + 1] = 0.0; // centroid 0, dim 1
        pq.centroids[cstart0 + 1 * dsub] = 0.0; // centroid 1, dim 0
        pq.centroids[cstart0 + 1 * dsub + 1] = 1.0; // centroid 1, dim 1

        // Sub-quantizer 1 (m=1): cstart = 1*256*2 = 512
        let cstart1 = 1 * ksub * dsub;
        pq.centroids[cstart1 + 0 * dsub] = 2.0; // centroid 0, dim 0
        pq.centroids[cstart1 + 0 * dsub + 1] = 0.0; // centroid 0, dim 1
        pq.centroids[cstart1 + 1 * dsub] = 0.0; // centroid 1, dim 0
        pq.centroids[cstart1 + 1 * dsub + 1] = 2.0; // centroid 1, dim 1

        pq
    }

    #[test]
    fn test_pq_compute_code_assigns_nearest_centroid() {
        // VAL-DICT-015: compute_code assigns nearest centroid
        let pq = make_known_pq();

        // x = [0.9, 0.1, 0.1, 1.9]
        // sub-vec 0 = [0.9, 0.1] → nearest to centroid 0 [1.0, 0.0]  (dist²≈0.02)
        // sub-vec 1 = [0.1, 1.9] → nearest to centroid 1 [0.0, 2.0]  (dist²≈0.02)
        let x = vec![0.9f32, 0.1, 0.1, 1.9];
        let mut code = vec![0u8; 2];
        pq.compute_code(&x, &mut code);
        assert_eq!(code[0], 0, "sub-vec 0 should map to centroid 0");
        assert_eq!(code[1], 1, "sub-vec 1 should map to centroid 1");
    }

    #[test]
    fn test_pq_compute_code_second_centroid() {
        // x = [0.1, 0.9, 1.9, 0.1]
        // sub-vec 0 = [0.1, 0.9] → nearest to centroid 1 [0.0, 1.0]
        // sub-vec 1 = [1.9, 0.1] → nearest to centroid 0 [2.0, 0.0]
        let pq = make_known_pq();
        let x = vec![0.1f32, 0.9, 1.9, 0.1];
        let mut code = vec![0u8; 2];
        pq.compute_code(&x, &mut code);
        assert_eq!(code[0], 1);
        assert_eq!(code[1], 0);
    }

    #[test]
    fn test_pq_compute_codes_all_rows() {
        // compute_codes should call compute_code for each row independently.
        let pq = make_known_pq();
        let data: Vec<f32> = vec![
            0.9, 0.1, 0.1, 1.9, // row 0 → codes [0, 1]
            0.1, 0.9, 1.9, 0.1, // row 1 → codes [1, 0]
        ];
        let n = 2i32;
        let mut codes = vec![0u8; 2 * pq.nsubq as usize];
        pq.compute_codes(&data, &mut codes, n);
        assert_eq!(codes[0], 0);
        assert_eq!(codes[1], 1);
        assert_eq!(codes[2], 1);
        assert_eq!(codes[3], 0);
    }

    // mulcode: dot product via centroid lookup

    #[test]
    fn test_pq_mulcode_matches_naive_dot() {
        // VAL-DICT-015: mulcode must match naive inner product within 1e-6
        let pq = make_known_pq();

        let x_data = vec![0.9f32, 0.1, 0.1, 1.9];
        let mut x = Vector::new(4);
        for (i, &v) in x_data.iter().enumerate() {
            x[i] = v;
        }
        // codes for row 0: code[0]=0, code[1]=1
        let codes: Vec<u8> = vec![0, 1];
        let alpha = 1.0f32;

        let result = pq.mulcode(&x, &codes, 0, alpha);

        // Naive: x[0..2] · c0_0 + x[2..4] · c1_1
        // c0_0 = [1.0, 0.0], c1_1 = [0.0, 2.0]
        // = (0.9*1.0 + 0.1*0.0) + (0.1*0.0 + 1.9*2.0)
        // = 0.9 + 3.8 = 4.7
        let expected = 0.9f32 * 1.0 + 0.1f32 * 0.0 + 0.1f32 * 0.0 + 1.9f32 * 2.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "mulcode={} expected={}",
            result,
            expected
        );
    }

    #[test]
    fn test_pq_mulcode_with_alpha() {
        let pq = make_known_pq();

        let x_data = vec![0.9f32, 0.1, 0.1, 1.9];
        let mut x = Vector::new(4);
        for (i, &v) in x_data.iter().enumerate() {
            x[i] = v;
        }
        let codes: Vec<u8> = vec![0, 1];
        let alpha = 2.5f32;

        let result = pq.mulcode(&x, &codes, 0, alpha);
        let base = 0.9f32 * 1.0 + 0.1f32 * 0.0 + 0.1f32 * 0.0 + 1.9f32 * 2.0; // 4.7
        let expected = base * alpha;
        assert!(
            (result - expected).abs() < 1e-5,
            "mulcode with alpha={}: got {} expected {}",
            alpha,
            result,
            expected
        );
    }

    #[test]
    fn test_pq_mulcode_multiple_rows() {
        // codes has two rows (nsubq=2 per row)
        let pq = make_known_pq();
        let x_data = vec![0.1f32, 0.9, 1.9, 0.1];
        let mut x = Vector::new(4);
        for (i, &v) in x_data.iter().enumerate() {
            x[i] = v;
        }
        // Row 0 codes: [1, 0], row 1 codes: [0, 1] (unused)
        let codes: Vec<u8> = vec![1, 0, 0, 1];
        let result = pq.mulcode(&x, &codes, 0, 1.0);
        // x[0..2] · c0_1 + x[2..4] · c1_0
        // c0_1=[0.0,1.0], c1_0=[2.0,0.0]
        // = (0.1*0 + 0.9*1) + (1.9*2 + 0.1*0) = 0.9 + 3.8 = 4.7
        let expected = 0.9f32 + 3.8f32;
        assert!((result - expected).abs() < 1e-6);
    }

    // addcode: add reconstructed row to vector

    #[test]
    fn test_pq_addcode_result() {
        // VAL-DICT-015: addcode result matches reconstruction
        let pq = make_known_pq();

        let mut x = Vector::new(4);
        // x starts at zero; addcode adds alpha * reconstructed row
        let codes: Vec<u8> = vec![0, 1]; // row 0: centroid 0 for sub-q 0, centroid 1 for sub-q 1
        pq.addcode(&mut x, &codes, 0, 1.0);

        // c0_0 = [1.0, 0.0], c1_1 = [0.0, 2.0]
        // x should be [1.0, 0.0, 0.0, 2.0]
        assert!((x[0] - 1.0).abs() < 1e-7);
        assert!((x[1] - 0.0).abs() < 1e-7);
        assert!((x[2] - 0.0).abs() < 1e-7);
        assert!((x[3] - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_pq_addcode_with_alpha() {
        let pq = make_known_pq();
        let mut x = Vector::new(4);
        let codes: Vec<u8> = vec![0, 1];
        pq.addcode(&mut x, &codes, 0, 3.0);
        assert!((x[0] - 3.0).abs() < 1e-6); // 3 * 1.0
        assert!((x[1] - 0.0).abs() < 1e-6); // 3 * 0.0
        assert!((x[2] - 0.0).abs() < 1e-6); // 3 * 0.0
        assert!((x[3] - 6.0).abs() < 1e-6); // 3 * 2.0
    }

    #[test]
    fn test_pq_addcode_accumulates() {
        // addcode should accumulate (not overwrite) existing vector values.
        let pq = make_known_pq();
        let mut x = Vector::new(4);
        x[0] = 5.0;
        x[3] = 1.0;
        let codes: Vec<u8> = vec![0, 1]; // adds [1.0, 0.0, 0.0, 2.0]
        pq.addcode(&mut x, &codes, 0, 1.0);
        assert!((x[0] - 6.0).abs() < 1e-7); // 5.0 + 1.0
        assert!((x[1] - 0.0).abs() < 1e-7); // 0.0 + 0.0
        assert!((x[2] - 0.0).abs() < 1e-7); // 0.0 + 0.0
        assert!((x[3] - 3.0).abs() < 1e-7); // 1.0 + 2.0
    }

    // mulcode vs addcode consistency

    #[test]
    fn test_pq_mulcode_addcode_consistency() {
        // mulcode(x, codes, t, 1) should equal the dot product of x with the
        // vector produced by addcode(zeros, codes, t, 1).
        let pq = make_known_pq();

        let x_data = vec![0.3f32, 0.7, 1.1, 0.5];
        let mut x = Vector::new(4);
        for (i, &v) in x_data.iter().enumerate() {
            x[i] = v;
        }
        let codes: Vec<u8> = vec![0, 1];

        // Compute mulcode result.
        let mul_result = pq.mulcode(&x, &codes, 0, 1.0);

        // Compute dot product with addcode reconstruction.
        let mut recon = Vector::new(4);
        pq.addcode(&mut recon, &codes, 0, 1.0);
        let dot: f32 = x_data
            .iter()
            .zip(recon.data().iter())
            .map(|(&a, &b)| a * b)
            .sum();

        assert!(
            (mul_result - dot).abs() < 1e-6,
            "mulcode={} dot={}",
            mul_result,
            dot
        );
    }

    // save / load round-trip

    #[test]
    fn test_pq_save_load_roundtrip() {
        // VAL-DICT-015: save/load must preserve all fields and centroids.
        let pq = make_known_pq();

        // Save.
        let mut buf = Vec::new();
        pq.save(&mut buf).expect("save should succeed");

        // Load.
        let mut cursor = Cursor::new(&buf);
        let pq2 = ProductQuantizer::load(&mut cursor).expect("load should succeed");

        // Fields must match.
        assert_eq!(pq2.dim, pq.dim);
        assert_eq!(pq2.nsubq, pq.nsubq);
        assert_eq!(pq2.dsub, pq.dsub);
        assert_eq!(pq2.lastdsub, pq.lastdsub);

        // Centroids must match exactly.
        assert_eq!(pq2.centroids.len(), pq.centroids.len());
        for (a, b) in pq.centroids.iter().zip(pq2.centroids.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "centroid mismatch");
        }
    }

    #[test]
    fn test_pq_save_load_odd_dim() {
        // Round-trip with odd dim decomposition.
        let mut pq = ProductQuantizer::new(11, 5);
        // Set some non-trivial centroids.
        for (i, v) in pq.centroids.iter_mut().enumerate() {
            *v = i as f32 * 0.001;
        }
        let mut buf = Vec::new();
        pq.save(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let pq2 = ProductQuantizer::load(&mut cursor).unwrap();

        assert_eq!(pq2.dim, 11);
        assert_eq!(pq2.nsubq, 3);
        assert_eq!(pq2.dsub, 5);
        assert_eq!(pq2.lastdsub, 1);
        for (a, b) in pq.centroids.iter().zip(pq2.centroids.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_pq_save_byte_count() {
        // Save format: 4*i32 + dim*ksub*f32 bytes.
        let pq = ProductQuantizer::new(4, 2);
        let mut buf = Vec::new();
        pq.save(&mut buf).unwrap();
        let expected = 4 * 4 + 4 * 256 * 4; // 4 header i32s + dim*KSUB f32s
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn test_pq_save_load_preserves_compute_code() {
        // Loaded PQ should produce same codes as original.
        let pq = make_known_pq();
        let mut buf = Vec::new();
        pq.save(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let pq2 = ProductQuantizer::load(&mut cursor).unwrap();

        let x = vec![0.9f32, 0.1, 0.1, 1.9];
        let mut code1 = vec![0u8; 2];
        let mut code2 = vec![0u8; 2];
        pq.compute_code(&x, &mut code1);
        pq2.compute_code(&x, &mut code2);
        assert_eq!(code1, code2);
    }

    // Training smoke test (verify k-means runs without panic)

    #[test]
    fn test_pq_train_smoke() {
        // Train on synthetic data with enough rows; check centroids are non-zero.
        let dim = 10i32;
        let dsub = 5i32;
        let mut pq = ProductQuantizer::new(dim, dsub);

        // Create 300 rows of synthetic data (enough to run k-means).
        let n = 300i32;
        let mut data = vec![0.0f32; (n as usize) * (dim as usize)];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32).sin();
        }

        pq.train(n, &data);

        // At least some centroids should be non-zero after training.
        let non_zero = pq.centroids.iter().any(|&v| v != 0.0);
        assert!(non_zero, "centroids should be non-zero after training");
    }

    #[test]
    fn test_pq_train_too_few_rows() {
        // Training with fewer than KSUB rows should not panic.
        // When data is all-zero the centroids will also be all-zero (filled
        // from the training data, which happens to be zeros here).
        let mut pq = ProductQuantizer::new(4, 2);
        let data = vec![0.0f32; 4 * 4]; // only 4 rows, all zeros
        pq.train(4, &data); // should not panic
        // Centroids are filled from zero data, so all remain 0.0.
        assert!(pq.centroids.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pq_train_small_n() {
        // Train PQ with n=10 data points (fewer than KSUB=256).
        // Verifies:
        //   1. Training completes without panic.
        //   2. Centroids are filled with training data (non-zero).
        //   3. compute_code produces valid codes for each training point.
        //   4. Distinct training points receive distinct codes.
        //   5. Each training point is reconstructed from its own centroid
        //      (code yields the point back via addcode within tolerance).
        let dim = 4i32;
        let dsub = 2i32;
        let n = 10i32;
        let mut pq = ProductQuantizer::new(dim, dsub);

        // Build 10 distinct, non-zero training points.
        // training point i: [(i+1)*1, (i+1)*2, (i+1)*3, (i+1)*4]
        let mut data = vec![0.0f32; n as usize * dim as usize];
        for i in 0..n as usize {
            for j in 0..dim as usize {
                data[i * dim as usize + j] = (i as f32 + 1.0) * (j as f32 + 1.0);
            }
        }

        pq.train(n, &data);

        // 1. Centroids should be non-zero after training with non-zero data.
        assert!(
            pq.centroids.iter().any(|&v| v != 0.0),
            "centroids should be non-zero after small-n training"
        );

        // 2. compute_code must not panic and must return valid codes.
        let nsubq = pq.nsubq as usize;
        let mut all_codes: Vec<Vec<u8>> = Vec::new();
        for i in 0..n as usize {
            let xi = &data[i * dim as usize..(i + 1) * dim as usize];
            let mut code = vec![0u8; nsubq];
            pq.compute_code(xi, &mut code); // must not panic
            all_codes.push(code);
        }
        assert_eq!(all_codes.len(), n as usize);

        // 3. Distinct training points should receive distinct codes.
        // (Each point is its own centroid in the small-n case.)
        assert_ne!(
            all_codes[0], all_codes[1],
            "distinct training points should get distinct codes"
        );

        // 4. Each training point should be assigned code i for both sub-quantizers,
        //    since its sub-vector is the exact centroid at slot i.
        for i in 0..n as usize {
            for (m, &code_val) in all_codes[i].iter().enumerate() {
                assert_eq!(
                    code_val as usize, i,
                    "training point {} sub-quantizer {} should map to code {}",
                    i, m, i
                );
            }
        }

        // 5. addcode on the code for point i should reconstruct the sub-vectors
        //    of that training point exactly.
        for i in 0..n as usize {
            let mut reconstructed = Vector::new(dim as usize);
            pq.addcode(&mut reconstructed, &all_codes[i], 0, 1.0);
            let xi = &data[i * dim as usize..(i + 1) * dim as usize];
            for j in 0..dim as usize {
                assert!(
                    (reconstructed[j] - xi[j]).abs() < 1e-6,
                    "reconstruction mismatch at point {} dim {}: got {} expected {}",
                    i,
                    j,
                    reconstructed[j],
                    xi[j]
                );
            }
        }
    }

    // get_centroids accessor consistency

    #[test]
    fn test_pq_get_centroids_matches_direct_slice() {
        let pq = make_known_pq();
        // Check that get_centroids(0, 0) == [1.0, 0.0]
        let c = pq.get_centroids(0, 0);
        assert_eq!(c.len(), 2);
        assert!((c[0] - 1.0).abs() < 1e-9);
        assert!((c[1] - 0.0).abs() < 1e-9);

        // get_centroids(0, 1) == [0.0, 1.0]
        let c = pq.get_centroids(0, 1);
        assert!((c[0] - 0.0).abs() < 1e-9);
        assert!((c[1] - 1.0).abs() < 1e-9);

        // get_centroids(1, 1) == [0.0, 2.0]
        let c = pq.get_centroids(1, 1);
        assert!((c[0] - 0.0).abs() < 1e-9);
        assert!((c[1] - 2.0).abs() < 1e-9);
    }
}
