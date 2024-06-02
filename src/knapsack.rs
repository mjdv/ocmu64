use std::{
    mem::swap,
    ops::{Add, AddAssign, SubAssign},
};

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

impl SubAssign for P {
    fn sub_assign(&mut self, other: Self) {
        *self = P(self.0 - other.0, self.1 - other.1);
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

#[derive(Default, Debug)]
pub struct KnapsackCache {
    points: Vec<P>,
    front: Vec<P>,
    new_front: Vec<P>,
}

/// Given:
/// - a target (-c,-c)
/// - a list of points (xi, yi)
/// returns whether there is a subset of points with sum <= target.
pub fn knapsack(
    mut target: P,
    points_it: impl Iterator<Item = P> + Clone,
    mut non_empty: bool,
    cache: &mut KnapsackCache,
) -> bool {
    // Sum of negative x.
    let mut neg_x = 0;
    // Sum of negative y.
    let mut neg_y = 0;
    // Sum of points <= (0,0).
    let mut sum = P(0, 0);

    for p in points_it.clone() {
        neg_x += p.0.min(0);
        neg_y += p.1.min(0);
        if p <= P(0, 0) {
            sum += p;
            non_empty = false;
            if sum <= target {
                return true;
            }
        }
    }

    if neg_x > target.0 || neg_y > target.1 {
        return false;
    }

    if sum <= target && !non_empty {
        return true;
    }

    target -= sum;

    let points = &mut cache.points;
    points.clear();
    points_it.filter(|&p| !(p <= P(0, 0))).collect_into(points);

    // the lowest point in Q2.
    let mut pq2 = P(0, 1);
    // the leftmost point in Q4.
    let mut pq4 = P(1, 0);

    for &p in &*points {
        if p <= target {
            return true;
        }
        if p.0 < 0 {
            assert!(p.1 > 0);
            if p.0 * pq2.1 < p.1 * pq2.0 {
                pq2 = p;
            }
        }
        if p.1 < 0 {
            assert!(p.0 > 0);
            if p.1 * pq4.0 < p.0 * pq4.1 {
                pq4 = p;
            }
        }
    }

    // All points are in a halfplane not containing the target.
    if pq2.0 * pq4.1 <= pq2.1 * pq4.0 {
        if target <= P(0, 0) {
            return false;
        }
        if target.1 > 0 {
            if pq2.0 * target.1 > pq2.1 * target.0 {
                return false;
            }
        }
        if target.0 > 0 {
            if pq4.1 * target.0 > pq4.0 * target.1 {
                return false;
            }
        }
    }

    // Sort points by sum of coordinates.
    points.sort_unstable_by_key(|&P(x, y)| x + y);

    // eprintln!("target: {target:?}");
    // eprintln!("pq2:    {pq2:?}");
    // eprintln!("pq4:    {pq4:?}");
    // eprintln!("points: {points:?}");

    let front = &mut cache.front;
    let new_front = &mut cache.new_front;
    front.clear();
    new_front.clear();

    front.push(P(0, 0));

    for &p in &*points {
        let l = front.len();
        let mut fi = 0;

        // Merge fronts into new_front.
        new_front.clear();
        new_front.reserve(2 * front.len());
        for i in 0..l {
            let sum = front[i] + p;
            if sum <= target {
                return true;
            }
            while fi < front.len() && (front[fi].0, front[fi].1) <= (sum.0, sum.1) {
                new_front.push(front[fi]);
                fi += 1;
            }
            new_front.push(sum);
        }
        new_front.extend_from_slice(&front[fi..]);

        swap(front, new_front);

        // Simplify front.
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

    // eprintln!(
    //     "NO BLOCKING SET after {} steps front size {}",
    //     points.len(),
    //     front.len()
    // );

    false
}
