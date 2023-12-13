use std::{
    fmt::Display,
    iter::Step,
    ops::{Deref, Index, IndexMut},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NodeA(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NodeB(pub usize);

impl Display for NodeA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}A", self.0)
    }
}

impl Display for NodeB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}B", self.0)
    }
}

impl Step for NodeA {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(|x| NodeA(x))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(|x| NodeA(x))
    }
}

impl Step for NodeB {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(|x| NodeB(x))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(|x| NodeB(x))
    }
}

#[derive(Default, Debug)]
pub struct VecA<T> {
    pub(crate) v: Vec<T>,
}

#[derive(Default, Debug, Clone)]
pub struct VecB<T> {
    pub(crate) v: Vec<T>,
}

impl<T> Deref for VecA<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> Deref for VecB<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> Index<NodeA> for VecA<T> {
    type Output = T;
    fn index(&self, index: NodeA) -> &Self::Output {
        &self.v[index.0]
    }
}

impl<T> IndexMut<NodeA> for VecA<T> {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        &mut self.v[index.0]
    }
}

impl<T> Index<NodeB> for VecB<T> {
    type Output = T;
    fn index(&self, index: NodeB) -> &Self::Output {
        &self.v[index.0]
    }
}

impl<T> IndexMut<NodeB> for VecB<T> {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.v[index.0]
    }
}
