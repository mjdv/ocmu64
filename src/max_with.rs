pub trait MaxWith {
    fn max_with(&mut self, other: Self) -> &mut Self;
    fn min_with(&mut self, other: Self) -> &mut Self;
}

impl<T: Ord + Copy> MaxWith for T {
    fn max_with(&mut self, other: Self) -> &mut Self {
        *self = std::cmp::Ord::max(*self, other);
        self
    }
    fn min_with(&mut self, other: Self) -> &mut Self {
        *self = std::cmp::Ord::min(*self, other);
        self
    }
}
