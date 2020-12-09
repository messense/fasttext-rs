#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use std::ffi::CStr;
use std::os::raw::c_char;

mod binding;

pub use self::binding::*;

pub fn error_message(ptr: *mut c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let s = format!("{}", c_str.to_string_lossy());
    unsafe {
        cft_str_free(ptr);
    }
    s
}

#[macro_export]
macro_rules! ffi_try {
    ($func:ident($($arg:expr),*)) => ({
        use std::ptr;
        let mut err = ptr::null_mut();
        let res = $crate::$func($($arg),*, &mut err);
        if !err.is_null() {
            return Err($crate::error_message(err));
        }
        res
    })
}
