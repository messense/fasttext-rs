
use std::io::{Read, Write};

use crate::error::Result;

/// FNV-1a hash function matching C++ fastText `Dictionary::hash` exactly.
///
/// Critical: each byte is cast as `i8` (signed) before widening to `u32` for the XOR step,
/// matching the C++ behavior `h = h ^ uint32_t(int8_t(str[i]))`. This means bytes > 127
/// are sign-extended (e.g., 0xC0 becomes 0xFFFFFF40 as u32, not 0xC0).
///
/// Algorithm: FNV-1a with offset basis 2166136261 and prime 16777619.
pub fn hash(s: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &byte in s {
        // Cast byte as i8 (signed), then widen to u32.
        // In Rust, `byte as i8` reinterprets the bits as signed.
        // Then `as i32 as u32` sign-extends to 32 bits, matching C++ `uint32_t(int8_t(b))`.
        h ^= byte as i8 as i32 as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Read a little-endian `i32` from a reader.
pub fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a little-endian `i64` from a reader.
pub fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a little-endian `f32` from a reader.
pub fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a little-endian `f64` from a reader.
pub fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Write a little-endian `i32` to a writer.
pub fn write_i32<W: Write>(writer: &mut W, value: i32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Write a little-endian `i64` to a writer.
pub fn write_i64<W: Write>(writer: &mut W, value: i64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Write a little-endian `f32` to a writer.
pub fn write_f32<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Write a little-endian `f64` to a writer.
pub fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Read a boolean (1 byte) from a reader.
pub fn read_bool<R: Read>(reader: &mut R) -> Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

/// Write a boolean (1 byte) to a writer.
pub fn write_bool<W: Write>(writer: &mut W, value: bool) -> Result<()> {
    writer.write_all(&[value as u8])?;
    Ok(())
}

/// Apply softmax normalization in-place over `data[..len]`.
///
/// Uses max-subtraction for numerical stability, matching the C++ fastText
/// softmax used in `SoftmaxLoss::compute_output` and the quantized prediction path.
pub fn softmax_in_place(data: &mut [f32], len: usize) {
    let slice = &mut data[..len];
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for v in slice.iter_mut() {
        *v = (*v - max).exp();
        z += *v;
    }
    if z > 0.0 {
        let inv_z = 1.0 / z;
        for v in slice.iter_mut() {
            *v *= inv_z;
        }
    }
}

/// L2-normalize a slice in-place. If the norm is below `1e-8`, the slice is left unchanged.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return;
    }
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Wrapper for `f32` that implements `Ord` (NaN-safe, treats NaN as equal).
///
/// Used for ordering f32 values in binary heaps and sorted collections
/// throughout the codebase (top-k prediction, nearest-neighbor search, etc.).
#[derive(Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helper: round-trip a value through write then read
    fn roundtrip_i32(val: i32) -> i32 {
        let mut buf = Vec::new();
        write_i32(&mut buf, val).unwrap();
        let mut cursor = Cursor::new(&buf);
        read_i32(&mut cursor).unwrap()
    }

    fn roundtrip_i64(val: i64) -> i64 {
        let mut buf = Vec::new();
        write_i64(&mut buf, val).unwrap();
        let mut cursor = Cursor::new(&buf);
        read_i64(&mut cursor).unwrap()
    }

    fn roundtrip_f32(val: f32) -> f32 {
        let mut buf = Vec::new();
        write_f32(&mut buf, val).unwrap();
        let mut cursor = Cursor::new(&buf);
        read_f32(&mut cursor).unwrap()
    }

    fn roundtrip_f64(val: f64) -> f64 {
        let mut buf = Vec::new();
        write_f64(&mut buf, val).unwrap();
        let mut cursor = Cursor::new(&buf);
        read_f64(&mut cursor).unwrap()
    }

    #[test]
    fn test_binary_io_roundtrip_i32() {
        assert_eq!(roundtrip_i32(0), 0);
        assert_eq!(roundtrip_i32(-1), -1);
        assert_eq!(roundtrip_i32(1), 1);
        assert_eq!(roundtrip_i32(i32::MAX), i32::MAX);
        assert_eq!(roundtrip_i32(i32::MIN), i32::MIN);
        assert_eq!(roundtrip_i32(42), 42);
        assert_eq!(roundtrip_i32(-42), -42);
        assert_eq!(roundtrip_i32(793712314), 793712314); // fastText magic number
    }

    #[test]
    fn test_binary_io_roundtrip_i64() {
        assert_eq!(roundtrip_i64(0), 0);
        assert_eq!(roundtrip_i64(-1), -1);
        assert_eq!(roundtrip_i64(1), 1);
        assert_eq!(roundtrip_i64(i64::MAX), i64::MAX);
        assert_eq!(roundtrip_i64(i64::MIN), i64::MIN);
        assert_eq!(roundtrip_i64(42), 42);
        assert_eq!(roundtrip_i64(-42), -42);
    }

    #[test]
    fn test_binary_io_roundtrip_f32() {
        assert_eq!(roundtrip_f32(0.0), 0.0);
        assert_eq!(roundtrip_f32(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(roundtrip_f32(1.0), 1.0);
        assert_eq!(roundtrip_f32(-1.0), -1.0);
        assert_eq!(roundtrip_f32(f32::MAX), f32::MAX);
        assert_eq!(roundtrip_f32(f32::MIN), f32::MIN);
        assert_eq!(roundtrip_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(roundtrip_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(roundtrip_f32(f32::MIN_POSITIVE), f32::MIN_POSITIVE);
        assert!(roundtrip_f32(f32::NAN).is_nan());
        // Subnormal value
        let subnormal: f32 = 1.0e-40;
        assert_eq!(roundtrip_f32(subnormal), subnormal);
    }

    #[test]
    fn test_binary_io_roundtrip_f64() {
        assert_eq!(roundtrip_f64(0.0), 0.0);
        assert_eq!(roundtrip_f64(-0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(roundtrip_f64(1.0), 1.0);
        assert_eq!(roundtrip_f64(-1.0), -1.0);
        assert_eq!(roundtrip_f64(f64::MAX), f64::MAX);
        assert_eq!(roundtrip_f64(f64::MIN), f64::MIN);
        assert_eq!(roundtrip_f64(f64::INFINITY), f64::INFINITY);
        assert_eq!(roundtrip_f64(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(roundtrip_f64(f64::MIN_POSITIVE), f64::MIN_POSITIVE);
        assert!(roundtrip_f64(f64::NAN).is_nan());
        // Subnormal value
        let subnormal: f64 = 5.0e-324;
        assert_eq!(roundtrip_f64(subnormal), subnormal);
        // Test 1e-4 (common in fastText as t parameter)
        assert_eq!(roundtrip_f64(1e-4), 1e-4);
    }

    #[test]
    fn test_binary_io_little_endian_layout() {
        // Verify that the bytes are actually written in little-endian order
        let mut buf = Vec::new();
        write_i32(&mut buf, 0x04030201).unwrap();
        assert_eq!(buf, vec![0x01, 0x02, 0x03, 0x04]);

        let mut buf = Vec::new();
        write_i64(&mut buf, 0x0807060504030201).unwrap();
        assert_eq!(buf, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_binary_io_write_read_sequence() {
        // Write multiple values in sequence and read them back
        let mut buf = Vec::new();
        write_i32(&mut buf, 100).unwrap();
        write_i64(&mut buf, 200).unwrap();
        write_f32(&mut buf, 3.14).unwrap();
        write_f64(&mut buf, 2.71828).unwrap();

        let mut cursor = Cursor::new(&buf);
        assert_eq!(read_i32(&mut cursor).unwrap(), 100);
        assert_eq!(read_i64(&mut cursor).unwrap(), 200);
        assert!((read_f32(&mut cursor).unwrap() - 3.14).abs() < 1e-6);
        assert!((read_f64(&mut cursor).unwrap() - 2.71828).abs() < 1e-10);
    }

    #[test]
    fn test_binary_io_read_truncated() {
        // Reading from too-short buffer should return an error
        let buf = vec![0u8; 2];
        let mut cursor = Cursor::new(&buf);
        assert!(read_i32(&mut cursor).is_err());

        let buf = vec![0u8; 4];
        let mut cursor = Cursor::new(&buf);
        assert!(read_i64(&mut cursor).is_err());

        let buf = vec![0u8; 2];
        let mut cursor = Cursor::new(&buf);
        assert!(read_f32(&mut cursor).is_err());

        let buf = vec![0u8; 4];
        let mut cursor = Cursor::new(&buf);
        assert!(read_f64(&mut cursor).is_err());
    }

    #[test]
    fn test_binary_io_roundtrip_primitives() {
        // VAL-CORE-013: comprehensive round-trip test for all primitives including edge values
        // i32 edge values
        for &val in &[0i32, -1, 1, i32::MAX, i32::MIN, 42, -42, 793712314] {
            assert_eq!(roundtrip_i32(val), val, "i32 roundtrip failed for {}", val);
        }
        // i64 edge values
        for &val in &[0i64, -1, 1, i64::MAX, i64::MIN, 42, -42] {
            assert_eq!(roundtrip_i64(val), val, "i64 roundtrip failed for {}", val);
        }
        // f32 edge values
        for &val in &[0.0f32, -1.0, 1.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE] {
            assert_eq!(roundtrip_f32(val), val, "f32 roundtrip failed for {}", val);
        }
        assert_eq!(roundtrip_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(roundtrip_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(roundtrip_f32(f32::NAN).is_nan());

        // f64 edge values
        for &val in &[0.0f64, -1.0, 1.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE] {
            assert_eq!(roundtrip_f64(val), val, "f64 roundtrip failed for {}", val);
        }
        assert_eq!(roundtrip_f64(f64::INFINITY), f64::INFINITY);
        assert_eq!(roundtrip_f64(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(roundtrip_f64(f64::NAN).is_nan());

        // Very small subnormals
        let subnormal_f32: f32 = 1.0e-40;
        assert_eq!(roundtrip_f32(subnormal_f32), subnormal_f32);
        let subnormal_f64: f64 = 5.0e-324;
        assert_eq!(roundtrip_f64(subnormal_f64), subnormal_f64);
    }

    // FNV-1a hash tests (VAL-CORE-005, VAL-CORE-006, VAL-CORE-007)

    #[test]
    fn test_fnv1a_reference_vectors() {
        // VAL-CORE-005: FNV-1a hash matches C++ reference vectors
        assert_eq!(hash(b""), 2166136261, "hash('') should be FNV offset basis");
        assert_eq!(hash(b"hello"), 1335831723);
        assert_eq!(hash(b"a"), 3826002220);
        assert_eq!(hash(b"</s>"), 3617362777);
    }

    #[test]
    fn test_fnv1a_signed_byte_extension() {
        // VAL-CORE-006: bytes > 127 must be sign-extended (int8_t cast)
        // For bytes 0-127, signed and unsigned produce the same result.
        // For bytes 128-255, they must differ because of sign extension.

        // Helper: compute hash with unsigned (wrong) path for comparison
        fn hash_unsigned(s: &[u8]) -> u32 {
            let mut h: u32 = 2166136261;
            for &byte in s {
                h ^= byte as u32; // unsigned cast (NOT what C++ fastText does)
                h = h.wrapping_mul(16777619);
            }
            h
        }

        // Single byte 0x80: signed=-128, unsigned=128
        let h_signed = hash(&[0x80]);
        let h_unsigned = hash_unsigned(&[0x80]);
        assert_ne!(
            h_signed, h_unsigned,
            "0x80 should differ between signed and unsigned paths"
        );
        assert_eq!(h_signed, 83079839, "hash([0x80]) with signed extension");

        // Single byte 0xFF: signed=-1, unsigned=255
        let h_signed = hash(&[0xFF]);
        let h_unsigned = hash_unsigned(&[0xFF]);
        assert_ne!(
            h_signed, h_unsigned,
            "0xFF should differ between signed and unsigned paths"
        );
        assert_eq!(h_signed, 4193493326, "hash([0xFF]) with signed extension");

        // Single byte 0xFE: signed=-2, unsigned=254
        let h_signed = hash(&[0xFE]);
        let h_unsigned = hash_unsigned(&[0xFE]);
        assert_ne!(
            h_signed, h_unsigned,
            "0xFE should differ between signed and unsigned paths"
        );
        assert_eq!(h_signed, 4210270945, "hash([0xFE]) with signed extension");

        // Single byte 0xC0: signed=-64, unsigned=192
        let h_signed = hash(&[0xC0]);
        let h_unsigned = hash_unsigned(&[0xC0]);
        assert_ne!(
            h_signed, h_unsigned,
            "0xC0 should differ between signed and unsigned paths"
        );
        assert_eq!(h_signed, 3304279519, "hash([0xC0]) with signed extension");
    }

    #[test]
    fn test_fnv1a_utf8_multibyte() {
        // VAL-CORE-007: hash of multi-byte UTF-8 strings must match C++ output
        // C++ processes raw bytes, not Unicode code points, so we pass raw UTF-8 bytes.
        assert_eq!(hash("日本語".as_bytes()), 308035559);
        assert_eq!(hash("café".as_bytes()), 1970454601);
    }

    #[test]
    fn test_fnv1a_empty_string() {
        // Empty string should return the FNV offset basis
        assert_eq!(hash(b""), 2166136261);
    }

    #[test]
    fn test_fnv1a_single_byte_values() {
        // Verify all 256 single-byte values: bytes 0-127 should match unsigned path,
        // bytes 128-255 should differ from unsigned path.
        fn hash_unsigned(s: &[u8]) -> u32 {
            let mut h: u32 = 2166136261;
            for &byte in s {
                h ^= byte as u32;
                h = h.wrapping_mul(16777619);
            }
            h
        }

        for b in 0u8..=127 {
            assert_eq!(
                hash(&[b]),
                hash_unsigned(&[b]),
                "byte {} (0x{:02X}) should match between signed and unsigned for values 0-127",
                b,
                b
            );
        }

        for b in 128u8..=255 {
            assert_ne!(
                hash(&[b]),
                hash_unsigned(&[b]),
                "byte {} (0x{:02X}) should differ between signed and unsigned for values 128-255",
                b,
                b
            );
        }
    }

    #[test]
    fn test_fnv1a_longer_strings() {
        // Verify determinism and that different strings produce different hashes
        let h1 = hash(b"test");
        let h2 = hash(b"test");
        assert_eq!(h1, h2, "Same input must produce same hash");

        let h3 = hash(b"tset");
        assert_ne!(h1, h3, "Different inputs should produce different hashes");

        // Multi-word strings
        let h4 = hash(b"the quick brown fox");
        let h5 = hash(b"the quick brown fox");
        assert_eq!(h4, h5, "Same longer input must produce same hash");
    }

    #[test]
    fn test_fnv1a_hash_str_convenience() {
        // Verify that passing &str as bytes works correctly
        let s = "hello";
        assert_eq!(hash(s.as_bytes()), 1335831723);
    }
}
