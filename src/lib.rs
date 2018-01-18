extern crate cfasttext_sys;

use std::ffi::CString;
use cfasttext_sys::*;

#[derive(Debug, Clone)]
pub struct FastText {
    inner: fasttext_t
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
