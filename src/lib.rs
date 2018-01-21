extern crate cfasttext_sys;

use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int};
use std::slice;
use std::borrow::Cow;

use cfasttext_sys::*;

#[derive(Debug, Clone)]
pub struct Args {
    inner: *mut fasttext_args_t
}

#[derive(Debug, Clone)]
pub struct FastText {
    inner: *mut fasttext_t
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub prob: f32,
    pub label: String,
}

impl Args {
    pub fn new() -> Self {
        unsafe {
            Self {
                inner: cft_args_new()
            }
        }
    }
    pub fn parse<T: AsRef<str>>(self, args: &[T]) {
        let argv: Vec<CString> = args.iter().map(|s| CString::new(s.as_ref()).unwrap()).collect();
        // FIXME: cft_fasttext_train should take *const *const c_char?
        let mut c_argv: Vec<*const c_char> = argv.iter().map(|s| s.as_ptr()).collect();
        unsafe {
            cft_args_parse(self.inner, c_argv.len() as c_int, c_argv.as_mut_ptr() as *mut *mut _ as *mut *mut _);
        }
    }

    pub fn input(&self) -> Cow<str> {
        unsafe {
            let ret = cft_args_get_input(self.inner);
            CStr::from_ptr(ret).to_string_lossy()
        }
    }

    pub fn set_input(&mut self, input: &str) {
        let c_input = CString::new(input).unwrap();
        unsafe {
            cft_args_set_input(self.inner, c_input.as_ptr());
        }
    }

    pub fn output(&self) -> Cow<str> {
        unsafe {
            let ret = cft_args_get_output(self.inner);
            CStr::from_ptr(ret).to_string_lossy()
        }
    }

    pub fn set_output(&mut self, input: &str) {
        let c_input = CString::new(input).unwrap();
        unsafe {
            cft_args_set_output(self.inner, c_input.as_ptr());
        }
    }
}

impl Drop for Args {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                cft_args_free(self.inner);
            }
        }
    }
}

impl Default for Args {
    fn default() -> Self {
        Args::new()
    }
}

impl Default for FastText {
    fn default() -> Self {
        FastText::new()
    }
}

impl FastText {
    pub fn new() -> Self {
        unsafe {
            Self {
                inner: cft_fasttext_new()
            }
        }
    }

    pub fn load_model(&mut self, filename: &str) {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            cft_fasttext_load_model(self.inner, c_path.as_ptr());
        }
    }

    pub fn save_model(&mut self) {
        unsafe {
            cft_fasttext_save_model(self.inner);
        }
    }

    pub fn save_output(&mut self) {
        unsafe {
            cft_fasttext_save_output(self.inner);
        }
    }

    pub fn save_vectors(&mut self) {
        unsafe {
            cft_fasttext_save_vectors(self.inner);
        }
    }

    pub fn get_dimension(&self) -> isize {
        unsafe {
            cft_fasttext_get_dimension(self.inner) as isize
        }
    }

    pub fn get_word_id(&self, word: &str) -> isize {
        let c_word = CString::new(word).unwrap();
        unsafe {
            cft_fasttext_get_word_id(self.inner, c_word.as_ptr()) as isize
        }
    }

    pub fn get_subword_id(&self, word: &str) -> isize {
        let c_word = CString::new(word).unwrap();
        unsafe {
            cft_fasttext_get_subword_id(self.inner, c_word.as_ptr()) as isize
        }
    }

    pub fn is_quant(&self) -> bool {
        unsafe {
            cft_fasttext_is_quant(self.inner)
        }
    }

    pub fn analogies(&mut self, k: i32) {
        unsafe {
            cft_fasttext_analogies(self.inner, k);
        }
    }

    pub fn train_thread(&mut self, n: u32) {
        unsafe {
            cft_fasttext_train_thread(self.inner, n as i32);
        }
    }

    pub fn load_vectors(&mut self, filename: &str) {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            cft_fasttext_load_vectors(self.inner, c_path.as_ptr());
        }
    }

    pub fn train(&mut self, args: &Args) {
        unsafe {
            cft_fasttext_train(self.inner, args.inner);
        }
    }

    pub fn predict(&self, text: &str, k: i32, threshold: f32) -> Vec<Prediction> {
        let c_text = CString::new(text).unwrap();
        unsafe {
            let ret = cft_fasttext_predict(self.inner, c_text.as_ptr(), k, threshold);
            let c_preds = slice::from_raw_parts((*ret).predictions, (*ret).length);
            let preds: Vec<Prediction> = c_preds.iter().map(|p| {
                let label = CStr::from_ptr((*p).label).to_string_lossy().to_string();
                Prediction {
                    prob: (*p).prob,
                    label: label
                }
            }).collect();
            cft_fasttext_predictions_free(ret);
            preds
        }
    }

    pub fn quantize(&mut self, args: &Args) {
        unsafe {
            cft_fasttext_quantize(self.inner, args.inner);
        }
    }
}

impl Drop for FastText {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                cft_fasttext_free(self.inner);
            }
        }
    }
}

unsafe impl Send for FastText {}
unsafe impl Sync for FastText {}

#[cfg(test)]
mod tests {
    use super::{Args, FastText};

    #[test]
    fn test_args_new_default() {
        let _args = Args::default();
    }

    #[test]
    fn test_args_input() {
        let mut args = Args::new();
        args.set_input("input");
        assert_eq!("input", args.input());
    }

    #[test]
    fn test_args_output() {
        let mut args = Args::new();
        args.set_output("output.model");
        assert_eq!("output.model", args.output());
    }

    #[test]
    fn test_fasttest_new_default() {
        let _fasttext = FastText::default();
    }
}
