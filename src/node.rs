use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    iter::Step,
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
};

pub trait NodeTrait: Copy + PartialEq + Eq + PartialOrd + Ord + Default + Serialize {
    const CHAR: char;
    const T: PhantomData<Self> = PhantomData;
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize)]
pub struct Node<NT>(pub usize, pub PhantomData<NT>);

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize,
)]
pub struct AT;
impl NodeTrait for AT {
    const CHAR: char = 'A';
}
#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize,
)]
pub struct BT;
impl NodeTrait for BT {
    const CHAR: char = 'B';
}

pub type NodeA = Node<AT>;
pub type NodeB = Node<BT>;

/// A function to create a NodeA.
#[allow(non_snake_case)]
pub fn NodeA(v: usize) -> NodeA {
    Node(v, PhantomData)
}
/// A function to create a NodeB.
#[allow(non_snake_case)]
pub fn NodeB(v: usize) -> NodeB {
    Node(v, PhantomData)
}

impl<NT: NodeTrait> Debug for Node<NT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.0, NT::CHAR)
    }
}

impl<NT: NodeTrait> Display for Node<NT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.0, NT::CHAR)
    }
}

impl<NT: NodeTrait> Step for Node<NT> {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(|x| Self(x, PhantomData))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(|x| Self(x, PhantomData))
    }
}

#[derive(Default, Debug, Clone)]
pub struct NodeVec<T, NT> {
    pub(crate) v: Vec<T>,
    _marker: PhantomData<NT>,
}

pub type VecA<T> = NodeVec<T, AT>;
pub type VecB<T> = NodeVec<T, BT>;

/// Vector of A nodes.
/// Indexing is unchecked by default.
impl<NT: NodeTrait, T: Default + Clone> NodeVec<T, NT> {
    pub fn from(v: Vec<T>) -> Self {
        NodeVec {
            v,
            _marker: PhantomData,
        }
    }
    pub fn new(a: Node<NT>) -> Self {
        NodeVec {
            v: vec![T::default(); a.0],
            _marker: PhantomData,
        }
    }
    pub fn len(&self) -> Node<NT> {
        Node(self.v.len(), PhantomData)
    }
    pub fn push(&mut self) -> Node<NT> {
        let id = self.len();
        self.v.push(T::default());
        id
    }
}
impl<NT1: NodeTrait, NT2: NodeTrait> NodeVec<Vec<Node<NT2>>, NT1> {
    pub fn nb_len(&self) -> Node<NT2> {
        Step::forward(
            *self.iter().filter_map(|x| x.iter().max()).max().unwrap(),
            1,
        )
    }
}

impl<NT: NodeTrait, T> Deref for NodeVec<T, NT> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<NT: NodeTrait, T> DerefMut for NodeVec<T, NT> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

/// Unchecked indexing.
impl<NT: NodeTrait, T> Index<Node<NT>> for NodeVec<T, NT> {
    type Output = T;
    fn index(&self, index: Node<NT>) -> &Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            self.v.get_unchecked(index.0)
        }
        #[cfg(debug_assertions)]
        &self.v[index.0]
    }
}

/// Unchecked indexing.
impl<NT: NodeTrait, T> IndexMut<Node<NT>> for NodeVec<T, NT> {
    fn index_mut(&mut self, index: Node<NT>) -> &mut Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            self.v.get_unchecked_mut(index.0)
        }
        #[cfg(debug_assertions)]
        &mut self.v[index.0]
    }
}
