use std::io::{Read, Write};

use crate::error::Result;

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
            assert_eq!(
                roundtrip_f32(val),
                val,
                "f32 roundtrip failed for {}",
                val
            );
        }
        assert_eq!(roundtrip_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(roundtrip_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(roundtrip_f32(f32::NAN).is_nan());

        // f64 edge values
        for &val in &[0.0f64, -1.0, 1.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE] {
            assert_eq!(
                roundtrip_f64(val),
                val,
                "f64 roundtrip failed for {}",
                val
            );
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
}
