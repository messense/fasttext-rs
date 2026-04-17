use std::io;

/// Error type for all fastText operations.
#[derive(Debug, thiserror::Error)]
pub enum FastTextError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    /// An invalid argument was provided.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    /// The model file is invalid or corrupted.
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    /// A NaN value was encountered during computation.
    #[error("Encountered NaN")]
    EncounteredNaN,
}

/// A specialized `Result` type for fastText operations.
pub type Result<T> = std::result::Result<T, FastTextError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io;

    #[test]
    fn test_error_types() {
        // Verify all variants can be constructed
        let _io = FastTextError::IoError(io::Error::new(io::ErrorKind::NotFound, "not found"));
        let _arg = FastTextError::InvalidArgument("bad arg".to_string());
        let _model = FastTextError::InvalidModel("bad model".to_string());
        let _nan = FastTextError::EncounteredNaN;
    }

    #[test]
    fn test_error_display() {
        let io_err =
            FastTextError::IoError(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        let display = format!("{}", io_err);
        assert!(!display.is_empty());
        assert!(display.contains("I/O error"));
        assert!(display.contains("file not found"));

        let arg_err = FastTextError::InvalidArgument("bad argument".to_string());
        let display = format!("{}", arg_err);
        assert!(display.contains("Invalid argument"));
        assert!(display.contains("bad argument"));

        let model_err = FastTextError::InvalidModel("corrupt header".to_string());
        let display = format!("{}", model_err);
        assert!(display.contains("Invalid model"));
        assert!(display.contains("corrupt header"));

        let nan_err = FastTextError::EncounteredNaN;
        let display = format!("{}", nan_err);
        assert!(display.contains("NaN"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let ft_err: FastTextError = io_err.into();
        match &ft_err {
            FastTextError::IoError(e) => {
                assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_error_source() {
        let io_err = FastTextError::IoError(io::Error::new(io::ErrorKind::NotFound, "not found"));
        assert!(io_err.source().is_some());

        let arg_err = FastTextError::InvalidArgument("bad".to_string());
        assert!(arg_err.source().is_none());

        let model_err = FastTextError::InvalidModel("bad".to_string());
        assert!(model_err.source().is_none());

        let nan_err = FastTextError::EncounteredNaN;
        assert!(nan_err.source().is_none());
    }

    #[test]
    fn test_error_debug() {
        let err = FastTextError::EncounteredNaN;
        let debug = format!("{:?}", err);
        assert!(!debug.is_empty());
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }
        fn returns_err() -> Result<i32> {
            Err(FastTextError::InvalidArgument("test".to_string()))
        }
        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }
}
