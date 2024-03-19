use crate::knapsack::{knapsack, P};

use super::*;
use std::cmp::max;

#[derive(Debug, Default, Clone)]
pub struct GraphBuilder {
    pub a: NodeA,
    pub b: NodeB,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
    /// The number of crossings from merging twins.
    pub self_crossings: u64,

    /// The inverse to map a solution for the reduced graph back to the original graph.
    pub inverse: Inverse,
}

/// Maps the B indices of a solution for the reduced graph back to the original graph.
type Inverse = VecB<Vec<NodeB>>;
fn new_inverse(b: NodeB) -> Inverse {
    VecB::from((NodeB(0)..b).map(|x| vec![x]).collect())
}

impl Graph {
    pub fn builder(&self) -> GraphBuilder {
        GraphBuilder {
            a: self.a,
            b: self.b,
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            self_crossings: self.self_crossings,
            inverse: new_inverse(self.b),
        }
    }
}

impl GraphBuilder {
    pub fn with_sizes(a: NodeA, b: NodeB) -> Self {
        Self {
            a,
            b,
            connections_a: VecA::new(a),
            connections_b: VecB::new(b),
            self_crossings: 0,
            inverse: new_inverse(b),
        }
    }
    /// Returns the id of the pushed node (not the length).
    pub fn push_node_a(&mut self) -> NodeA {
        let id = self.connections_a.push();
        self.a = self.connections_a.len();
        id
    }

    /// Returns the id of the pushed node (not the length).
    pub fn push_node_b(&mut self) -> NodeB {
        let id = self.connections_b.push();
        self.b = self.connections_b.len();
        assert_eq!(self.inverse.len(), id);
        self.inverse.push();
        self.inverse[id] = vec![id];
        id
    }

    pub fn push_edge(&mut self, a: NodeA, b: NodeB) {
        self[a].push(b);
        self[b].push(a);
    }

    pub fn try_push_edge(&mut self, a: NodeA, b: NodeB) -> bool {
        if self[a].contains(&b) {
            false
        } else {
            self.push_edge(a, b);
            true
        }
    }

    pub fn new(connections_b: VecB<Vec<NodeA>>) -> GraphBuilder {
        let inv = (NodeB(0)..connections_b.len())
            .map(|x| vec![x])
            .collect_vec();
        Self::new_with_inv(connections_b, &inv)
    }

    pub fn new_with_inv(connections_b: VecB<Vec<NodeA>>, inv: &[Vec<NodeB>]) -> GraphBuilder {
        let a = Step::forward(
            *connections_b
                .iter()
                .filter_map(|x| x.iter().max())
                .max()
                .unwrap(),
            1,
        );
        let b = connections_b.len();
        let mut g = Self {
            a,
            b,
            connections_a: Default::default(),
            connections_b,
            self_crossings: 0,
            inverse: VecB::from(inv.to_vec()),
        };
        g.reconstruct_a();
        g
    }

    pub fn to_quick_graph(&self) -> Graph {
        let (crossings, reduced_crossings) = self.crossings();
        Graph {
            a: self.a,
            b: self.b,
            m: self.connections_a.iter().map(|x| x.len()).sum::<usize>(),
            b_permutation: Default::default(),
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            crossings,
            reduced_crossings,
            intervals: self.intervals(),
            self_crossings: self.self_crossings,
            before: VecB::new(self.b),
        }
    }

    /// Split into non-trivial graphs corresponding to disjoint intervals.
    fn split(&self) -> Vec<GraphBuilder> {
        let mut intervals = self.intervals();
        // Suffix min of start
        {
            let mut min_start = self.a;
            for r in intervals.iter_mut().rev() {
                min_start = min(min_start, r.start);
                r.start = min_start;
            }
        }
        // Prefix max of end
        {
            let mut max_end = NodeA(0);
            for r in intervals.iter_mut() {
                max_end = max(max_end, r.end);
                r.end = max_end;
            }
        }

        let mut graphs = vec![];
        let mut i = 0;
        let mut last_a_range: Option<(NodeA, NodeA)> = None;
        while i < self.b.0 {
            // Find the component starting at intervals[i].
            let mut j = i + 1;
            let mut end = intervals[NodeB(i)].end;
            while j < self.b.0 && intervals[NodeB(j)].start < end {
                end = max(end, intervals[NodeB(j)].end);
                j += 1;
            }
            let start = intervals[NodeB(i)].start.0;
            // Convert the component into a Graph on its own.
            let mut new_cb = self.connections_b.v[i..j].to_vec();
            let new_inv = self.inverse.v[i..j].to_vec();

            // Make sure the new range is disjoint with the previous one and that a split is allowed here.
            let new_a_range = new_cb
                .iter()
                .flat_map(|x| x.iter())
                .cloned()
                .minmax()
                .into_option();
            if let Some(new_a_range) = new_a_range {
                if let Some(last_a_range) = last_a_range {
                    assert2::assert!(last_a_range.1 <= new_a_range.0);
                }
                last_a_range = Some(new_a_range);
            }

            // Adjust A coordinates.
            for l in &mut new_cb {
                for x in l {
                    assert!(start <= x.0);
                    *x = NodeA(x.0 - start);
                }
            }
            let g = GraphBuilder::new_with_inv(VecB::from(new_cb), &new_inv);
            graphs.push(g);
            i = j;
        }
        info!(
            "Split into parts with sizes: {:?}",
            graphs.iter().map(|g| (g.a, g.b)).collect::<Vec<_>>()
        );
        graphs
    }

    pub fn num_edges(&self) -> usize {
        self.connections_a.iter().map(|x| x.len()).sum::<usize>()
    }

    pub fn to_graph(&mut self) -> Graph {
        let (crossings, reduced_crossings) = self.crossings();
        let mut before = self.dominating_pairs();
        self.practical_dominating_pairs(&mut before);
        self.boundary_pairs(&mut before);
        Self::transitive_closure(&mut before);

        Graph {
            a: self.a,
            b: self.b,
            m: self.connections_a.iter().map(|x| x.len()).sum::<usize>(),
            b_permutation: Default::default(),
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            crossings,
            reduced_crossings,
            intervals: self.intervals(),
            self_crossings: self.self_crossings,
            before,
        }
    }

    /// Print statistics on the graph.
    fn print_stats(&self) {
        info!("A={:?} B={:?}, {} edges", self.a, self.b, self.num_edges());

        if !log::log_enabled!(log::Level::Debug) {
            return;
        }

        let degs = self.connections_b.iter().map(|x| x.len()).join(",");
        debug!("Degrees:\n{degs}");

        draw(&self.connections_b, true);
    }

    /// Permute the vertices of the graph so that B is roughly increasing.
    /// TODO: Store metadata to be able to invert the final solution.
    /// TODO: Allow customizing whether crossings are built.
    pub fn build(&mut self) -> Vec<GraphBuilder> {
        self.drop_singletons();
        if get_flag("no_transform") {
            return vec![self.clone()];
        }
        self.merge_twins();
        self.merge_adjacent_edges();
        self.sort_edges();
        self.permute(initial_solution(&self.to_quick_graph()));
        self.sort_edges();
        self.print_stats();
        if get_flag("no_split") {
            let copy = self.clone();
            self.self_crossings = 0;
            vec![copy]
        } else {
            self.split()
        }
    }

    fn sort_edges(&mut self) {
        for l in self.connections_a.iter_mut() {
            l.sort();
        }
        for l in self.connections_b.iter_mut() {
            l.sort();
        }
    }

    /// Drop all nodes of degree 0.
    fn drop_singletons(&mut self) {
        let a_old = self.a;
        let b_old = self.b;
        {
            let mut b = NodeB(0);
            let mut dropped: Vec<NodeB> = vec![];
            self.inverse.retain(|inv| {
                let retain = !self.connections_b[b].is_empty();
                if !retain {
                    dropped.extend(inv);
                }
                b = Step::forward(b, 1);
                retain
            });
            self.inverse[NodeB(0)].extend(dropped);
        }
        self.connections_b.retain(|b| !b.is_empty());
        self.reconstruct_a();
        self.connections_a.retain(|a| !a.is_empty());
        self.reconstruct_b();
        self.sort_edges();
        let da = Step::steps_between(&self.a, &a_old).unwrap();
        let db = Step::steps_between(&self.b, &b_old).unwrap();
        if da > 0 || db > 0 {
            info!("Dropped {da} and {db} singletons",);
        }
    }

    /// Merge vertices with the same set of neighbours.
    fn merge_twins(&mut self) {
        self.sort_edges();
        let mut perm = (NodeB(0)..self.b).collect_vec();
        perm.sort_by_key(|&x| &self.connections_b[x]);
        let inv_len = self.inverse.iter().flatten().count();
        let mut self_crossings = 0;
        assert_eq!(self.b, self.inverse.len());
        let (new_b, new_inv) = (0..self.b.0)
            .group_by(|x| self.connections_b[perm[*x]].clone())
            .into_iter()
            .map(|(key, group)| {
                let mut cnt = 0;
                let new_inverse = group
                    .inspect(|_| cnt += 1)
                    .flat_map(|x| &self.inverse[perm[x]])
                    .cloned()
                    .collect();
                let pairs = cnt * (cnt - 1) / 2;
                let incr = Self::edge_list_crossings(&key, &key) * pairs as u64;
                self_crossings += incr;
                (
                    key.iter()
                        .flat_map(|x| std::iter::repeat(*x).take(cnt))
                        .collect(),
                    new_inverse,
                )
            })
            .unzip();
        self.connections_b = VecB::from(new_b);
        self.inverse = VecB::from(new_inv);
        let new_inv_len = self.inverse.iter().flatten().count();
        assert_eq!(inv_len, new_inv_len);
        self.self_crossings += self_crossings;
        info!(
            "Merged {} twins; {self_crossings} self crossings",
            Step::steps_between(&self.connections_b.len(), &self.b).unwrap()
        );
        self.reconstruct_a();
    }

    /// Given two adjacent nodes (u,v) in A.
    /// If the nbs of u and v are *only* connected to u and v, then we can merge u and v, and we can merge their neighbours.
    fn merge_adjacent_edges(&mut self) {
        self.sort_edges();

        let mut merged = 0;
        for ((x, cl), (y, cr)) in (NodeA(0)..self.a)
            .zip(self.connections_a.iter())
            .filter(|(_, cl)| !cl.is_empty())
            .tuple_windows()
        {
            if cl.is_empty() || cr.is_empty() {
                continue;
            }
            // Single nb each.
            if cl.first() != cl.last() {
                continue;
            }
            if cr.first() != cr.last() {
                continue;
            }

            let u = cl.first().unwrap();
            let v = cr.first().unwrap();

            if self[*u].is_empty() || self[*v].is_empty() {
                continue;
            }

            // Those nbs must only connect to cl and cr.
            if self[*u].first() != self[*u].last() {
                continue;
            }
            if self[*v].first() != self[*v].last() {
                continue;
            }

            assert_eq!(self[*u][0], x);
            assert_eq!(self[*v][0], y);

            // Merge u into v.
            assert!(self[y].len() == self[*v].len());
            let l = self[*u].len();
            self.connections_b[*u].clear();
            self.connections_b[*v].extend((0..l).map(|_| y));
            // NOTE: We preserve order of the inverse mapping.
            // Since u is left of v, the preimage of u must come before the preimage of v.
            let mut inv_u = std::mem::take(&mut self.inverse[*u]);
            let inv_v = std::mem::take(&mut self.inverse[*v]);
            inv_u.extend(inv_v);
            self.inverse[*v] = inv_u;
            merged += 1;
        }

        info!("Merged {merged} adjacent edges",);
        self.reconstruct_a();
        self.drop_singletons();
    }

    /// Find pairs (u,v) with equal degree and neighbours(u) <= neighbours(v).
    fn dominating_pairs(&mut self) -> Before {
        let mut before = Before::from(vec![VecB::from(vec![Unordered; self.b.0]); self.b.0]);

        let mut disjoint_pairs = 0;

        // Set the trivial ones.
        for u in NodeB(0)..self.b {
            for v in NodeB(0)..self.b {
                if self[u].last() < self[v].first() {
                    before[u][v] = Before;
                    disjoint_pairs += 1;
                }
                if self[u].first() > self[v].last() {
                    before[u][v] = After;
                    disjoint_pairs += 1;
                }
            }
        }
        info!("Found {disjoint_pairs} disjoint pairs");

        if get_flag("no_dominating_pairs") {
            return before;
        }

        let mut dominating_pairs = 0;

        self.sort_edges();
        for u in NodeB(0)..self.b {
            for v in NodeB(0)..self.b {
                if u == v || before[u][v] != Unordered || self[u] == self[v] {
                    continue;
                }

                if (0..self[u].len()).all(|i| {
                    // u[i] must be smaller than v[j]
                    // for all j>=floor(i*vl/ul)
                    let j = (i * self[v].len()) / self[u].len();
                    self[u][i] <= self[v][j]
                }) {
                    before[u][v] = Before;
                    before[v][u] = After;
                    dominating_pairs += 1;
                }
            }
        }
        info!("Found {dominating_pairs} dominating pairs");
        before
    }

    fn practical_dominating_pairs(&mut self, before: &mut Before) {
        if get_flag("no_pd") {
            info!("Found 0 practical dominating pairs (skipped)");
            return;
        }

        let mut practical_dominating_pairs = 0;
        let mut not_practical_dominating_pairs = 0;

        self.sort_edges();

        let cr = self.crossings().1;

        let xs = (NodeB(0)..self.b).collect_vec();

        for u in NodeB(0)..self.b {
            for v in NodeB(0)..self.b {
                match is_practically_dominating_pair(u, v, before, &cr, &xs) {
                    IsPDP::Skip => {}
                    IsPDP::No => {
                        not_practical_dominating_pairs += 1;
                    }
                    IsPDP::Yes => {
                        practical_dominating_pairs += 1;
                        before[u][v] = Before;
                        before[v][u] = After;
                    }
                }
            }
        }
        info!("Dropped {not_practical_dominating_pairs} candidate practical dominating pairs");
        info!("Found {practical_dominating_pairs} practical dominating pairs");
    }
}

pub enum IsPDP {
    // Equivalent to No but not counted.
    Skip,
    No,
    Yes,
}

// Is u forced before v because there is no separating set?
pub fn is_practically_dominating_pair(
    u: NodeB,
    v: NodeB,
    before: &Before,
    cr: &ReducedCrossings,
    xs: &[NodeB],
) -> IsPDP {
    if u == v || before[v][u] != Unordered {
        return IsPDP::Skip;
    }

    // We want to prove that u < v.

    // Try to find a set X such that vXu is the strictly optimal rotation.
    // If no such set exists, then u < v.

    // TODO: Better handle this equality case. We have to be careful
    // to not introduce cycles.
    if cr[v][u] <= 0 {
        return IsPDP::Skip;
    }

    // knapsack
    let target = P(cr[u][v] as i32, cr[u][v] as i32);

    // We do not consider x that must be before u and v, or after u and v.
    let points = xs.iter().filter_map(|&x| {
        if x == u || x == v {
            return None;
        }
        // x must be left of u and v.
        if before[u][x] == After && before[v][x] == After {
            return None;
        }
        // x must be right of u and v.
        if before[u][x] == Before && before[v][x] == Before {
            return None;
        }
        // x that want to be between u and v (as in uxv) are not useful here.
        if cr[u][x] <= 0 && cr[x][v] <= 0 {
            return None;
        }
        Some(P(cr[x][u] as i32, cr[v][x] as i32))
    });

    if !knapsack(target, points) {
        IsPDP::Yes
    } else {
        IsPDP::No
    }
}

impl GraphBuilder {
    /// When a cell is green and there is not a single red cell left-below it, fix the order of the pair.
    /// I.e.: When u < v in the current order, and for all (x,y) with x <= u < v <= y we want x < y, then fix u < v.
    fn boundary_pairs(&mut self, before: &mut Before) {
        if !get_flag("boundary_pairs") {
            info!("Found 0 boundary pairs (skipped)");
            return;
        }

        let eq_boundary_pairs = get_flag("eq_boundary_pairs");

        let mut boundary_pairs = 0;

        self.sort_edges();

        let mut leftmost_red = self.b;

        for v in (NodeB(0)..self.b).rev() {
            leftmost_red = min(leftmost_red, v);

            let mut pending = None;

            for u in NodeB(0)..leftmost_red {
                if before[v][u] != Unordered {
                    continue;
                }
                // TODO: Also handle equality cases properly.
                if self.one_node_crossings(u, v) == self.one_node_crossings(v, u) {
                    if pending.is_none() {
                        pending = Some(u);
                    }
                } else if self.one_node_crossings(u, v) < self.one_node_crossings(v, u) {
                    if let Some(pending) = pending
                        && eq_boundary_pairs
                    {
                        for u in pending..u {
                            before[u][v] = Before;
                            before[v][u] = After;
                            boundary_pairs += 1;
                        }
                    }
                    pending = None;
                    before[u][v] = Before;
                    before[v][u] = After;
                    boundary_pairs += 1;
                } else {
                    leftmost_red = u;
                    break;
                }
            }
        }

        info!("Found {boundary_pairs} boundary pairs");
    }

    // This is quite slow.
    // TODO: Does it contribute anything at all anyway?
    fn transitive_closure(before: &mut Before) {
        if !get_flag("tc") {
            return;
        }
        let mut changed = true;
        let mut transitive_pairs = 0;
        while changed {
            changed = false;
            for i in NodeB(0)..before.len() {
                for j in NodeB(0)..before.len() {
                    if before[i][j] == Before {
                        for k in NodeB(0)..before.len() {
                            if before[j][k] == Before && before[i][k] != Before {
                                before[i][k] = Before;
                                before[k][j] = After;
                                changed = true;
                                transitive_pairs += 1;
                            }
                        }
                    }
                }
            }
            info!("Found {transitive_pairs} transitive pairs");
        }
    }

    /// Reconstruct `connections_a`, given `connections_b`.
    pub fn reconstruct_a(&mut self) {
        self.a = self.connections_b.nb_len();
        self.b = self.connections_b.len();
        self.connections_a = VecA::new(self.a);
        for b in NodeB(0)..self.b {
            for &a in self.connections_b[b].iter() {
                self.connections_a[a].push(b);
            }
        }
    }

    /// Reconstruct `connections_b`, given `connections_a`.
    fn reconstruct_b(&mut self) {
        self.a = self.connections_a.len();
        self.b = self.connections_a.nb_len();
        self.connections_b = VecB::new(self.b);
        for a in NodeA(0)..self.a {
            for &b in self.connections_a[a].iter() {
                self.connections_b[b].push(a);
            }
        }
    }

    /// Permute the nodes of B such that the given solution is simply 0..b.
    fn permute(&mut self, solution: Solution) {
        self.inverse = VecB::from(
            solution
                .iter()
                .map(|&b| std::mem::take(&mut self.inverse[b]))
                .collect(),
        );
        self.connections_b = VecB::from(
            solution
                .iter()
                .map(|&b| std::mem::take(&mut self.connections_b[b]))
                .collect(),
        );
        self.reconstruct_a();
    }

    fn intervals(&self) -> VecB<Range<NodeA>> {
        VecB::from(
            self.connections_b
                .iter()
                .map(|b| {
                    let (min, max) = b
                        .iter()
                        .copied()
                        .minmax()
                        .into_option()
                        .unwrap_or((NodeA(0), NodeA(0)));
                    min..max
                })
                .collect(),
        )
    }

    pub fn edge_list_crossings(e1: &Vec<NodeA>, e2: &Vec<NodeA>) -> u64 {
        let mut result: u64 = 0;
        for edge_i in e1 {
            for edge_j in e2 {
                if edge_i > edge_j {
                    result += 1;
                }
            }
        }
        result
    }

    pub fn one_node_crossings(&self, i: NodeB, j: NodeB) -> u64 {
        Self::edge_list_crossings(&self[i], &self[j])
    }

    fn crossings(&self) -> (Crossings, ReducedCrossings) {
        let mut crossings = VecB::from(vec![VecB::new(self.b); self.b.0]);
        for node_i in NodeB(0)..self.b {
            for node_j in NodeB(0)..self.b {
                let c = self.one_node_crossings(node_i, node_j);
                crossings[node_i][node_j] = c.try_into().unwrap_or_else(|_| {
                    panic!("Crossings between {node_i} and {node_j} is too large: {c}",)
                });
            }
        }
        let mut reduced_crossings = VecB::from(vec![VecB::new(self.b); self.b.0]);
        for i in NodeB(0)..self.b {
            for j in NodeB(0)..self.b {
                let cr = crossings[i][j] as i64 - crossings[j][i] as i64;
                reduced_crossings[i][j] = cr.try_into().unwrap_or_else(|_| {
                    panic!("Crossings between {i} and {j} is too large: {cr}",)
                });
            }
        }
        (crossings, reduced_crossings)
    }

    pub fn invert(&self, solution: Solution) -> Solution {
        assert_eq!(
            solution.len(),
            self.inverse.len().0,
            "Cannot invert solution with wrong length"
        );
        solution
            .iter()
            .flat_map(|&b| self.inverse[b].clone())
            .collect()
    }
}

impl Index<NodeA> for GraphBuilder {
    type Output = Vec<NodeB>;

    fn index(&self, index: NodeA) -> &Self::Output {
        &self.connections_a[index]
    }
}

impl IndexMut<NodeA> for GraphBuilder {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        &mut self.connections_a[index]
    }
}

impl Index<NodeB> for GraphBuilder {
    type Output = Vec<NodeA>;

    fn index(&self, index: NodeB) -> &Self::Output {
        &self.connections_b[index]
    }
}

impl IndexMut<NodeB> for GraphBuilder {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.connections_b[index]
    }
}
