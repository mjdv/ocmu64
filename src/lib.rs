#![feature(step_trait, is_sorted)]
pub mod generate;
pub mod graph;
pub mod node;

thread_local! {
    pub static FLAGS: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
}

pub fn get_flag(name: &str) -> bool {
    FLAGS.with(|f| f.borrow().iter().any(|flag| flag == name))
}
