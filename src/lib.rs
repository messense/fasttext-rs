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
pub enum ModelName {
    /// CBOW
    CBOW,
    /// SkipGram
    SG,
    /// Supervised
    SUP,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossName {
    HS,
    NS,
    SOFTMAX,
    OVA,
}

impl From<ModelName> for model_name_t {
    fn from(mt: ModelName) -> model_name_t {
        match mt {
            ModelName::CBOW => model_name_t::MODEL_CBOW,
            ModelName::SG => model_name_t::MODEL_SG,
            ModelName::SUP => model_name_t::MODEL_SUP,
        }
    }
}

impl From<model_name_t> for ModelName {
    fn from(mn: model_name_t) -> ModelName {
        match mn {
            model_name_t::MODEL_CBOW => ModelName::CBOW,
            model_name_t::MODEL_SG => ModelName::SG,
            model_name_t::MODEL_SUP => ModelName::SUP,
        }
    }
}

impl From<LossName> for loss_name_t {
    fn from(lt: LossName) -> loss_name_t {
        match lt {
            LossName::HS => loss_name_t::LOSS_HS,
            LossName::NS => loss_name_t::LOSS_NS,
            LossName::SOFTMAX => loss_name_t::LOSS_SOFTMAX,
            LossName::OVA => loss_name_t::LOSS_OVA,
        }
    }
}

impl From<loss_name_t> for LossName {
    fn from(ln: loss_name_t) -> LossName {
        match ln {
            loss_name_t::LOSS_HS => LossName::HS,
            loss_name_t::LOSS_NS => LossName::NS,
            loss_name_t::LOSS_SOFTMAX => LossName::SOFTMAX,
            loss_name_t::LOSS_OVA => LossName::OVA,
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

    pub fn model(&self) -> ModelName {
        let model_name = unsafe { cft_args_get_model(self.inner) };
        model_name.into()
    }

    pub fn set_model(&mut self, model: ModelName) {
        unsafe {
            cft_args_set_model(self.inner, model.into());
        }
    }

    pub fn loss(&self) -> LossName {
        let loss_name = unsafe { cft_args_get_loss(self.inner) };
        loss_name.into()
    }

    pub fn set_loss(&mut self, loss: LossName) {
        unsafe {
            cft_args_set_loss(self.inner, loss.into());
        }
    }

    pub fn min_count(&self) -> i32 {
        unsafe { cft_args_get_min_count(self.inner) }
    }

    pub fn set_min_count(&mut self, min_count: i32) {
        unsafe { cft_args_set_min_count(self.inner, min_count) }
    }

    pub fn min_count_label(&self) -> i32 {
        unsafe { cft_args_get_min_count_label(self.inner) }
    }

    pub fn set_min_count_label(&mut self, min_count: i32) {
        unsafe { cft_args_set_min_count_label(self.inner, min_count) }
    }

    pub fn neg(&self) -> i32 {
        unsafe { cft_args_get_neg(self.inner) }
    }

    pub fn set_neg(&mut self, neg: i32) {
        unsafe { cft_args_set_neg(self.inner, neg) }
    }

    pub fn word_ngrams(&self) -> i32 {
        unsafe { cft_args_get_word_ngrams(self.inner) }
    }

    pub fn set_word_ngrams(&mut self, ngrams: i32) {
        unsafe { cft_args_set_word_ngrams(self.inner, ngrams) }
    }

    pub fn bucket(&self) -> i32 {
        unsafe { cft_args_get_bucket(self.inner) }
    }

    pub fn set_bucket(&mut self, bucket: i32) {
        unsafe { cft_args_set_bucket(self.inner, bucket) }
    }

    pub fn minn(&self) -> i32 {
        unsafe { cft_args_get_minn(self.inner) }
    }

    pub fn set_minn(&mut self, minn: i32) {
        unsafe { cft_args_set_minn(self.inner, minn) }
    }

    pub fn maxn(&self) -> i32 {
        unsafe { cft_args_get_maxn(self.inner) }
    }

    pub fn set_maxn(&mut self, maxn: i32) {
        unsafe { cft_args_set_maxn(self.inner, maxn) }
    }

    pub fn t(&self) -> i32 {
        unsafe { cft_args_get_t(self.inner) }
    }

    pub fn set_t(&mut self, t: i32) {
        unsafe { cft_args_set_t(self.inner, t) }
    }

    pub fn verbose(&self) -> i32 {
        unsafe { cft_args_get_verbose(self.inner) }
    }

    pub fn set_verbose(&mut self, verbose: i32) {
        unsafe { cft_args_set_verbose(self.inner, verbose) }
    }

    pub fn label(&self) -> Cow<str> {
        unsafe {
            let ret = cft_args_get_label(self.inner);
            CStr::from_ptr(ret).to_string_lossy()
        }
    }

    pub fn set_label(&mut self, label: &str) {
        let c_label = CString::new(label).unwrap();
        unsafe {
            cft_args_set_label(self.inner, c_label.as_ptr());
        }
    }

    pub fn save_output(&self) -> bool {
        unsafe { cft_args_get_save_output(self.inner) }
    }

    pub fn set_save_output(&mut self, save_output: bool) {
        unsafe { cft_args_set_save_output(self.inner, save_output) }
    }

    pub fn qout(&self) -> bool {
        unsafe { cft_args_get_qout(self.inner) }
    }

    pub fn set_qout(&mut self, qout: bool) {
        unsafe { cft_args_set_qout(self.inner, qout) }
    }

    pub fn retrain(&self) -> bool {
        unsafe { cft_args_get_retrain(self.inner) }
    }

    pub fn set_retrain(&mut self, retrain: bool) {
        unsafe { cft_args_set_retrain(self.inner, retrain) }
    }

    pub fn qnorm(&self) -> bool {
        unsafe { cft_args_get_qnorm(self.inner) }
    }

    pub fn set_qnorm(&mut self, qnorm: bool) {
        unsafe { cft_args_set_qnorm(self.inner, qnorm) }
    }

    pub fn cutoff(&self) -> usize {
        unsafe { cft_args_get_cutoff(self.inner) }
    }

    pub fn set_cutoff(&mut self, cutoff: usize) {
        unsafe { cft_args_set_cutoff(self.inner, cutoff) }
    }

    pub fn dsub(&self) -> usize {
        unsafe { cft_args_get_dsub(self.inner) }
    }

    pub fn set_dsub(&mut self, dsub: usize) {
        unsafe { cft_args_set_dsub(self.inner, dsub) }
    }

    pub fn print_help(&self) {
        unsafe { cft_args_print_help(self.inner) }
    }

    pub fn print_basic_help(&self) {
        unsafe { cft_args_print_basic_help(self.inner) }
    }

    pub fn print_dictionary_help(&self) {
        unsafe { cft_args_print_dictionary_help(self.inner) }
    }

    pub fn print_training_help(&self) {
        unsafe { cft_args_print_training_help(self.inner) }
    }

    pub fn print_quantization_help(&self) {
        unsafe { cft_args_print_quantization_help(self.inner) }
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

    fn convert_predictions(c_preds: &[fasttext_prediction_t]) -> Vec<Prediction> {
        unsafe {
            let preds: Vec<Prediction> = c_preds.iter().map(|p| {
                let label = CStr::from_ptr((*p).label).to_string_lossy().to_string();
                Prediction {
                    prob: (*p).prob,
                    label: label
                }
            }).collect();
            preds
        }
    }

    pub fn predict(&self, text: &str, k: i32, threshold: f32) -> Vec<Prediction> {
        let c_text = CString::new(text).unwrap();
        unsafe {
            let ret = cft_fasttext_predict(self.inner, c_text.as_ptr(), k, threshold);
            let c_preds = slice::from_raw_parts((*ret).predictions, (*ret).length);
            let preds = FastText::convert_predictions(c_preds);
            cft_fasttext_predictions_free(ret);
            preds
        }
    }

    pub fn predict_on_words(&self, words: &Vec<i32>, k: i32, threshold: f32) -> Vec<Prediction> {
        unsafe {
            let mut ws: Vec<i32> = words.clone();
            let words = fasttext_words_t {words: ws.as_mut_ptr(), length: ws.len()};
            let ret = cft_fasttext_predict_on_words(self.inner, &words, k, threshold);
            let c_preds = slice::from_raw_parts((*ret).predictions, (*ret).length);
            let preds = FastText::convert_predictions(c_preds);
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
    use super::{Args, FastText, ModelName, LossName};

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
        assert_eq!(ModelName::SG, args.model());

        args.set_model(ModelName::CBOW);
        assert_eq!(ModelName::CBOW, args.model());
    }

    #[test]
    fn test_args_loss() {
        let mut args = Args::new();
        assert_eq!(LossName::NS, args.loss());

        args.set_loss(LossName::SOFTMAX);
        assert_eq!(LossName::SOFTMAX, args.loss());
    }

    #[test]
    fn test_args_lr() {
        let mut args = Args::new();
        assert_eq!(0.05, args.lr());

        args.set_lr(0.1);
        assert_eq!(0.1, args.lr());
    }

    #[test]
    fn test_args_lr_update_rate() {
        let mut args = Args::new();
        assert_eq!(100, args.lr_update_rate());

        args.set_lr_update_rate(50);
        assert_eq!(50, args.lr_update_rate());
    }

    #[test]
    fn test_args_dim() {
        let mut args = Args::new();
        assert_eq!(100, args.dim());

        args.set_dim(50);
        assert_eq!(50, args.dim());
    }

    #[test]
    fn test_args_ws() {
        let mut args = Args::new();
        assert_eq!(5, args.ws());

        args.set_ws(10);
        assert_eq!(10, args.ws());
    }

    #[test]
    fn test_args_epoch() {
        let mut args = Args::new();
        assert_eq!(5, args.epoch());

        args.set_epoch(50);
        assert_eq!(50, args.epoch());
    }

    #[test]
    fn test_args_thread() {
        let mut args = Args::new();
        args.set_thread(10);
        assert_eq!(10, args.thread());
    }

    #[test]
    fn test_args_min_count() {
        let mut args = Args::new();
        assert_eq!(5, args.min_count());

        args.set_min_count(10);
        assert_eq!(10, args.min_count());
    }

    #[test]
    fn test_args_min_count_label() {
        let mut args = Args::new();
        assert_eq!(0, args.min_count_label());

        args.set_min_count_label(10);
        assert_eq!(10, args.min_count_label());
    }

    #[test]
    fn test_args_neg() {
        let mut args = Args::new();
        assert_eq!(5, args.neg());

        args.set_neg(10);
        assert_eq!(10, args.neg());
    }

    #[test]
    fn test_args_word_ngrams() {
        let mut args = Args::new();
        assert_eq!(1, args.word_ngrams());

        args.set_word_ngrams(3);
        assert_eq!(3, args.word_ngrams());
    }

    #[test]
    fn test_args_bucket() {
        let mut args = Args::new();
        assert_eq!(2000000, args.bucket());

        args.set_bucket(1000000);
        assert_eq!(1000000, args.bucket());
    }

    #[test]
    fn test_args_minn() {
        let mut args = Args::new();
        assert_eq!(3, args.minn());

        args.set_minn(10);
        assert_eq!(10, args.minn());
    }

    #[test]
    fn test_args_maxn() {
        let mut args = Args::new();
        assert_eq!(6, args.maxn());

        args.set_maxn(10);
        assert_eq!(10, args.maxn());
    }

    #[test]
    fn test_args_t() {
        let mut args = Args::new();
        assert_eq!(0, args.t());

        args.set_t(10);
        assert_eq!(10, args.t());
    }

    #[test]
    fn test_args_verbose() {
        let mut args = Args::new();
        assert_eq!(2, args.verbose());

        args.set_verbose(1);
        assert_eq!(1, args.verbose());
    }

    #[test]
    fn test_args_label() {
        let mut args = Args::new();
        assert_eq!("__label__", args.label());

        args.set_label("__my_label__");
        assert_eq!("__my_label__", args.label());
    }

    #[test]
    fn test_args_save_output() {
        let mut args = Args::new();
        assert_eq!(false, args.save_output());

        args.set_save_output(true);
        assert_eq!(true, args.save_output());
    }

    #[test]
    fn test_args_qout() {
        let mut args = Args::new();
        assert_eq!(false, args.qout());

        args.set_qout(true);
        assert_eq!(true, args.qout());
    }

    #[test]
    fn test_args_retrain() {
        let mut args = Args::new();
        assert_eq!(false, args.retrain());

        args.set_retrain(true);
        assert_eq!(true, args.retrain());
    }

    #[test]
    fn test_args_qnorm() {
        let mut args = Args::new();
        assert_eq!(false, args.qnorm());

        args.set_qnorm(true);
        assert_eq!(true, args.qnorm());
    }

    #[test]
    fn test_args_cutoff() {
        let mut args = Args::new();
        assert_eq!(0, args.cutoff());

        args.set_cutoff(5);
        assert_eq!(5, args.cutoff());
    }

    #[test]
    fn test_args_dsub() {
        let mut args = Args::new();
        assert_eq!(2, args.dsub());

        args.set_dsub(5);
        assert_eq!(5, args.dsub());
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
