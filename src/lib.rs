#[macro_use]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    CBOW,
    SG,
    SUP,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    HS,
    NS,
    SOFTMAX,
}

impl From<ModelType> for model_name_t {
    fn from(mt: ModelType) -> model_name_t {
        match mt {
            ModelType::CBOW => model_name_t::MODEL_CBOW,
            ModelType::SG => model_name_t::MODEL_SG,
            ModelType::SUP => model_name_t::MODEL_SUP,
        }
    }
}

impl From<model_name_t> for ModelType {
    fn from(mn: model_name_t) -> ModelType {
        match mn {
            model_name_t::MODEL_CBOW => ModelType::CBOW,
            model_name_t::MODEL_SG => ModelType::SG,
            model_name_t::MODEL_SUP => ModelType::SUP,
        }
    }
}

impl From<LossType> for loss_name_t {
    fn from(lt: LossType) -> loss_name_t {
        match lt {
            LossType::HS => loss_name_t::LOSS_HS,
            LossType::NS => loss_name_t::LOSS_NS,
            LossType::SOFTMAX => loss_name_t::LOSS_SOFTMAX,
        }
    }
}

impl From<loss_name_t> for LossType {
    fn from(ln: loss_name_t) -> LossType {
        match ln {
            loss_name_t::LOSS_HS => LossType::HS,
            loss_name_t::LOSS_NS => LossType::NS,
            loss_name_t::LOSS_SOFTMAX => LossType::SOFTMAX,
        }
    }
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

    pub fn lr(&self) -> f64 {
        unsafe { cft_args_get_lr(self.inner) }
    }

    pub fn set_lr(&mut self, lr: f64) {
        unsafe { cft_args_set_lr(self.inner, lr) }
    }

    pub fn lr_update_rate(&self) -> i32 {
        unsafe { cft_args_get_lr_update_rate(self.inner) }
    }

    pub fn set_lr_update_rate(&mut self, rate: i32) {
        unsafe { cft_args_set_lr_update_rate(self.inner, rate) }
    }

    pub fn dim(&self) -> i32 {
        unsafe { cft_args_get_dim(self.inner) }
    }

    pub fn set_dim(&mut self, dim: i32) {
        unsafe { cft_args_set_dim(self.inner, dim) }
    }

    pub fn ws(&self) -> i32 {
        unsafe { cft_args_get_ws(self.inner) }
    }

    pub fn set_ws(&mut self, ws: i32) {
        unsafe { cft_args_set_ws(self.inner, ws) }
    }

    pub fn epoch(&self) -> i32 {
        unsafe { cft_args_get_epoch(self.inner) }
    }

    pub fn set_epoch(&mut self, epoch: i32) {
        unsafe { cft_args_set_epoch(self.inner, epoch) }
    }

    pub fn thread(&self) -> i32 {
        unsafe { cft_args_get_thread(self.inner) }
    }

    pub fn set_thread(&mut self, thread: i32) {
        unsafe { cft_args_set_thread(self.inner, thread) }
    }

    pub fn model(&self) -> ModelType {
        let model_name = unsafe { cft_args_get_model(self.inner) };
        model_name.into()
    }

    pub fn set_model(&mut self, model: ModelType) {
        unsafe {
            cft_args_set_model(self.inner, model.into());
        }
    }

    pub fn loss(&self) -> LossType {
        let loss_name = unsafe { cft_args_get_loss(self.inner) };
        loss_name.into()
    }

    pub fn set_loss(&mut self, loss: LossType) {
        unsafe {
            cft_args_set_loss(self.inner, loss.into());
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

    pub fn load_model(&mut self, filename: &str) -> Result<(), String> {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            ffi_try!(cft_fasttext_load_model(self.inner, c_path.as_ptr()));
        }
        Ok(())
    }

    pub fn save_model(&mut self, filename: &str) -> Result<(), String> {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            ffi_try!(cft_fasttext_save_model(self.inner, c_path.as_ptr()));
        }
        Ok(())
    }

    pub fn save_output(&mut self, filename: &str) -> Result<(), String> {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            ffi_try!(cft_fasttext_save_output(self.inner, c_path.as_ptr()));
        }
        Ok(())
    }

    pub fn save_vectors(&mut self, filename: &str) -> Result<(), String> {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            ffi_try!(cft_fasttext_save_vectors(self.inner, c_path.as_ptr()));
        }
        Ok(())
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

    pub fn load_vectors(&mut self, filename: &str) -> Result<(), String> {
        let c_path = CString::new(filename).unwrap();
        unsafe {
            ffi_try!(cft_fasttext_load_vectors(self.inner, c_path.as_ptr()));
        }
        Ok(())
    }

    pub fn train(&mut self, args: &Args) -> Result<(), String> {
        unsafe {
            ffi_try!(cft_fasttext_train(self.inner, args.inner));
        }
        Ok(())
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

    pub fn quantize(&mut self, args: &Args) -> Result<(), String> {
        unsafe {
            ffi_try!(cft_fasttext_quantize(self.inner, args.inner));
        }
        Ok(())
    }

    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let c_text = CString::new(word).unwrap();
        let dim = self.get_dimension() as usize;
        let mut v = Vec::with_capacity(dim);
        unsafe {
            cft_fasttext_get_word_vector(self.inner, c_text.as_ptr(), v.as_mut_ptr());
            v.set_len(dim);
        }
        v
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let c_text = CString::new(text).unwrap();
        unsafe {
            let ret = cft_fasttext_tokenize(self.inner, c_text.as_ptr());
            let c_tokens = slice::from_raw_parts((*ret).tokens, (*ret).length);
            let tokens: Vec<String> = c_tokens.iter().map(|p| {
                CStr::from_ptr(*p).to_string_lossy().to_string()
            }).collect();
            cft_fasttext_tokens_free(ret);
            tokens
        }
    }

    pub fn get_sentence_vector(&self, sentence: &str) -> Vec<f32> {
        let c_text = CString::new(sentence).unwrap();
        let dim = self.get_dimension() as usize;
        let mut v = Vec::with_capacity(dim);
        unsafe {
            cft_fasttext_get_sentence_vector(self.inner, c_text.as_ptr(), v.as_mut_ptr());
            v.set_len(dim);
        }
        v
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
    use super::{Args, FastText, ModelType, LossType};

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
    fn test_args_model() {
        let mut args = Args::new();
        assert_eq!(ModelType::SG, args.model());

        args.set_model(ModelType::CBOW);
        assert_eq!(ModelType::CBOW, args.model());
    }

    #[test]
    fn test_args_loss() {
        let mut args = Args::new();
        assert_eq!(LossType::NS, args.loss());

        args.set_loss(LossType::SOFTMAX);
        assert_eq!(LossType::SOFTMAX, args.loss());
    }

    #[test]
    fn test_fasttext_new_default() {
        let _fasttext = FastText::default();
    }

    #[test]
    fn test_fasttext_get_word_vector() {
        let mut fasttext = FastText::default();
        fasttext.load_model("tests/fixtures/cooking.model.bin").unwrap();
        
        // The model contains the word "banana", right?
        let v = fasttext.get_word_vector("banana");
        assert!(fasttext.get_dimension() == v.len() as isize);
        assert!(v[0] != 0f32);
        // And it doesn't contain "hello".
        assert!(fasttext.get_word_vector("hello")[0] == 0f32);
    }

    #[test]
    fn test_fasttext_get_sentence_vector() {
        let mut fasttext = FastText::default();
        fasttext.load_model("tests/fixtures/cooking.model.bin").unwrap();

        // The model contains the word "banana", right?
        let v = fasttext.get_sentence_vector("banana");
        assert!(fasttext.get_dimension() == v.len() as isize);
        assert!(v[0] != 0f32);
        // And it doesn't contain "hello".
        assert!(fasttext.get_sentence_vector("hello")[0] == 0f32);
    }

    #[test]
    fn test_fasttext_tokenize() {
        let fasttext = FastText::default();
        let tokens = fasttext.tokenize("I love banana");
        assert_eq!(tokens, ["I", "love", "banana"]);

        let tokens = fasttext.tokenize("不支持中文");
        assert_eq!(tokens, ["不支持中文"]);
    }
}
