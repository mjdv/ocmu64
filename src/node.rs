use std::{
    fmt::{Debug, Display},
    iter::Step,
    ops::{Deref, DerefMut, Index, IndexMut},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NodeA(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
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

impl Debug for NodeA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Debug for NodeB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Step for NodeA {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(NodeA)
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(NodeA)
    }
}

impl Step for NodeB {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(NodeB)
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(NodeB)
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

impl<T> VecA<T> {
    pub fn len(&self) -> NodeA {
        NodeA(self.v.len())
    }
}

impl<T> VecB<T> {
    pub fn len(&self) -> NodeB {
        NodeB(self.v.len())
    }
}

impl<T> Deref for VecA<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> DerefMut for VecA<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl<T> Deref for VecB<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> DerefMut for VecB<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
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
