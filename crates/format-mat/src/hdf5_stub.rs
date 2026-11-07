#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]

//! Minimal pure-Rust HDF5 Level 0 superblock reader.

use crate::FormatError;

/// The 8-byte HDF5 file signature.
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

/// Minimal HDF5 superblock header.
#[derive(Clone, Debug, PartialEq)]
pub struct HdfSuperblock {
    pub format_signature: [u8; 8],
    pub superblock_version: u8,
    pub base_address: u64,
}

/// Read a u64 from a byte slice at `pos` in little-endian order.
fn read_u64_le(data: &[u8], pos: usize) -> Result<u64, FormatError> {
    let slice = data
        .get(pos..pos.saturating_add(8))
        .ok_or_else(|| FormatError::MalformedMat("Unexpected EOF reading u64 in HDF5 superblock".into()))?;
    let arr: [u8; 8] = slice
        .try_into()
        .map_err(|_| FormatError::MalformedMat("Slice-to-array conversion failed".into()))?;
    Ok(u64::from_le_bytes(arr))
}

/// Parse the HDF5 superblock from raw bytes.
/// - Checks the 8-byte HDF5 signature (`\x89HDF\r\n\x1a\n`).
/// - Reads `superblock_version` byte at offset 8.
/// - Reads `base_address` as little-endian u64:
///   - Version 0 or 1: offset 24
///   - Version 2 or 3: offset 8
///
/// Returns `FormatError::MalformedMat("HDF5 signature mismatch")` on bad magic.
pub fn read_hdf5_superblock(data: &[u8]) -> Result<HdfSuperblock, FormatError> {
    if data.len() < 9 {
        return Err(FormatError::MalformedMat(
            "Data too short to contain HDF5 superblock".into(),
        ));
    }

    let sig_slice = data
        .get(..8)
        .ok_or_else(|| FormatError::MalformedMat("Cannot read HDF5 signature".into()))?;
    let sig: [u8; 8] = sig_slice
        .try_into()
        .map_err(|_| FormatError::MalformedMat("Signature slice conversion failed".into()))?;

    if sig != HDF5_SIGNATURE {
        return Err(FormatError::MalformedMat(
            "HDF5 signature mismatch".into(),
        ));
    }

    let superblock_version = data
        .get(8)
        .copied()
        .ok_or_else(|| FormatError::MalformedMat("Cannot read superblock_version".into()))?;

    let base_address = if superblock_version <= 1 {
        read_u64_le(data, 24)?
    } else {
        // Version 2/3: base_address at offset 8 per task spec
        read_u64_le(data, 8)?
    };

    Ok(HdfSuperblock {
        format_signature: sig,
        superblock_version,
        base_address,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hdf5_v0_header() -> Vec<u8> {
        let mut data = vec![0u8; 56];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0;
        let ba: u64 = 0x1234;
        data[24..32].copy_from_slice(&ba.to_le_bytes());
        data
    }

    #[test]
    fn test_read_hdf5_superblock_v0() {
        let data = make_hdf5_v0_header();
        let sb = read_hdf5_superblock(&data).unwrap();
        assert_eq!(sb.format_signature, HDF5_SIGNATURE);
        assert_eq!(sb.superblock_version, 0);
        assert_eq!(sb.base_address, 0x1234);
    }

    #[test]
    fn test_bad_signature_returns_error() {
        let mut data = make_hdf5_v0_header();
        data[0] = 0x00;
        let result = read_hdf5_superblock(&data);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("HDF5 signature mismatch"));
    }

    #[test]
    fn test_too_short_returns_error() {
        let result = read_hdf5_superblock(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_hdf5_superblock_v2() {
        let mut data = vec![0u8; 32];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 2;
        let sb = read_hdf5_superblock(&data).unwrap();
        assert_eq!(sb.superblock_version, 2);
        // base_address = u64 LE at offset 8 = 2 (version byte=2, rest zeros)
        assert_eq!(sb.base_address, 2);
    }
}
