#![feature(step_trait)]
pub mod database;
pub mod generate;
pub mod graph;
mod knapsack;
pub mod max_with;
pub mod node;
mod pattern_search;

thread_local! {
    pub static FLAGS: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
}

pub fn get_flag(name: &str) -> bool {
    FLAGS.with(|f| f.borrow().iter().any(|flag| flag == name))
}
pub fn clear_flags() {
    FLAGS.with(|f| f.borrow_mut().clear());
}
pub fn set_flags(flags: &[impl AsRef<str>]) {
    FLAGS.with(|f| {
        let mut f = f.borrow_mut();
        f.clear();
        f.extend(flags.iter().map(|f| f.as_ref().to_string()));
    });
}
