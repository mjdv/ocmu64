use std::{
    fmt::{Debug, Display},
    iter::Step,
    ops::{Deref, DerefMut, Index, IndexMut},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct NodeA(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
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

#[derive(Default, Debug, Clone)]
pub struct VecA<T> {
    pub(crate) v: Vec<T>,
}

#[derive(Default, Debug, Clone)]
pub struct VecB<T> {
    pub(crate) v: Vec<T>,
}

/// Vector of A nodes.
/// Indexing is unchecked by default.
impl<T: Default + Clone> VecA<T> {
    pub fn new(a: NodeA) -> Self {
        VecA {
            v: vec![T::default(); a.0],
        }
    }
    pub fn len(&self) -> NodeA {
        NodeA(self.v.len())
    }
    pub fn push(&mut self) -> NodeA {
        let id = self.len();
        self.v.push(T::default());
        id
    }
}

/// Vector of B nodes.
/// Indexing is unchecked by default.
impl<T: Default + Clone> VecB<T> {
    pub fn new(b: NodeB) -> Self {
        VecB {
            v: vec![T::default(); b.0],
        }
    }
    pub fn len(&self) -> NodeB {
        NodeB(self.v.len())
    }
    pub fn push(&mut self) -> NodeB {
        let id = self.len();
        self.v.push(T::default());
        id
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

/// Unchecked indexing.
impl<T> Index<NodeA> for VecA<T> {
    type Output = T;
    fn index(&self, index: NodeA) -> &Self::Output {
        #[cfg(release)]
        unsafe {
            self.v.get_unchecked(index.0)
        }
        #[cfg(not(release))]
        &self.v[index.0]
    }
}

/// Unchecked indexing.
impl<T> IndexMut<NodeA> for VecA<T> {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        #[cfg(release)]
        unsafe {
            self.v.get_unchecked_mut(index.0)
        }
        #[cfg(not(release))]
        &mut self.v[index.0]
    }
}

/// Unchecked indexing.
impl<T> Index<NodeB> for VecB<T> {
    type Output = T;
    fn index(&self, index: NodeB) -> &Self::Output {
        #[cfg(release)]
        unsafe {
            self.v.get_unchecked(index.0)
        }
        #[cfg(not(release))]
        &self.v[index.0]
    }
}

/// Unchecked indexing.
impl<T> IndexMut<NodeB> for VecB<T> {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        #[cfg(release)]
        unsafe {
            self.v.get_unchecked_mut(index.0)
        }
        #[cfg(not(release))]
        &mut self.v[index.0]
    }
}
