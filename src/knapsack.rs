use std::ops::{Add, AddAssign};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, Default)]
pub struct P(pub i32, pub i32);

impl Add for P {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        P(self.0 + other.0, self.1 + other.1)
    }
}

impl AddAssign for P {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl PartialOrd for P {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!();
    }

    fn le(&self, other: &Self) -> bool {
        self.0 <= other.0 && self.1 <= other.1
    }
}

/// Given:
/// - a target (-c,-c)
/// - a list of points (xi, yi)
/// returns whether there is a subset of points with sum <= target.
pub fn knapsack(target: P, points: &[P]) -> bool {
    // 1. Check if sum of negative x or sum of negative y is not small enough.
    if points.iter().map(|p| p.0.min(0)).sum::<i32>() > target.0
        || points.iter().map(|p| p.1.min(0)).sum::<i32>() > target.1
    {
        return false;
    }

    // 2: add all points <= (0,0).
    let mut sum = P(0, 0);
    for &p in points {
        if p <= P(0, 0) {
            sum += p;
        }
    }

    if sum <= target {
        return true;
    }

    let mut front = vec![sum];
    for &p in points {
        if p <= P(0, 0) {
            continue;
        }

        let l = front.len();
        front.reserve(l);
        for i in 0..l {
            let sum = front[i] + p;
            if sum <= target {
                return true;
            }
            front.push(sum);
        }

        // Simplify front.
        front.sort_by_key(|&P(x, y)| (x, y));

        let mut best = front[0];
        let mut first = true;
        front.retain(|&p| {
            if first {
                first = false;
                return true;
            }
            if best <= p {
                return false;
            }
            best = p;
            true
        });
    }

    false
}
