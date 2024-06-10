use crate::knapsack::{knapsack, KnapsackCache, P};

use super::*;
use std::cmp::max;

#[derive(Debug, Default, Clone)]
pub struct GraphBuilder {
    pub a_original: Option<NodeA>,
    pub a: NodeA,
    pub b: NodeB,
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

impl GraphBuilder {
    pub fn with_sizes(a: NodeA, b: NodeB) -> Self {
        Self {
            a_original: None,
            a,
            b,
            connections_b: VecB::new(b),
            self_crossings: 0,
            inverse: new_inverse(b),
        }
    }
    /// Returns the id of the pushed node (not the length).
    pub fn push_node_a(&mut self) -> NodeA {
        let id = self.a;
        self.a = Step::forward(self.a, 1);
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
        self[b].push(a);
    }

    pub fn try_push_edge(&mut self, a: NodeA, b: NodeB) -> bool {
        if self[b].contains(&a) {
            false
        } else {
            self.push_edge(a, b);
            true
        }
    }

    pub fn new(a: NodeA, connections_b: VecB<Vec<NodeA>>) -> GraphBuilder {
        let inv = (NodeB(0)..connections_b.len())
            .map(|x| vec![x])
            .collect_vec();
        Self::new_with_inv(Some(a), connections_b, &inv)
    }

    pub fn new_with_inv(
        a_original: Option<NodeA>,
        connections_b: VecB<Vec<NodeA>>,
        inv: &[Vec<NodeB>],
    ) -> GraphBuilder {
        let a = Step::forward(
            *connections_b
                .iter()
                .filter_map(|x| x.iter().max())
                .max()
                .unwrap(),
            1,
        );
        let b = connections_b.len();
        Self {
            a_original,
            a,
            b,
            connections_b,
            self_crossings: 0,
            inverse: VecB::from(inv.to_vec()),
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
            let g = GraphBuilder::new_with_inv(None, VecB::from(new_cb), &new_inv);
            graphs.push(g);
            i = j;
        }
        info!(
            "PARTS: {:?}",
            graphs.iter().map(|g| (g.a, g.b)).collect::<Vec<_>>()
        );
        graphs
    }

    pub fn num_edges(&self) -> usize {
        self.connections_b.iter().map(|x| x.len()).sum::<usize>()
    }

    pub fn to_graph(&mut self, full: bool) -> Graph {
        self.sort_edges();
        let (min_crossings, cr, cr_range, prefix_max, suffix_min) = self.crossings();

        let before = if full {
            let mut before = self.dominating_pairs(&cr_range);
            self.practical_dominating_pairs(&mut before, &cr);
            if get_flag("tc") {
                Self::transitive_closure(&mut before);
            }
            // TODO: Do PDP twice.
            // self.practical_dominating_pairs(&mut before, &cr);
            self.boundary_pairs(&mut before);
            before
        } else {
            VecB::new(self.b)
        };

        Graph {
            a: self.a,
            b: self.b,
            b_permutation: Default::default(),
            connections_b: self.connections_b.clone(),
            min_crossings,
            reduced_crossings: cr,
            cr_range,
            intervals: self.intervals(),
            prefix_max,
            suffix_min,
            self_crossings: self.self_crossings,
            before,
            knapsack_cache: KnapsackCache::default(),
        }
    }

    /// Print statistics on the graph.
    fn print_stats(&self) {
        info!(
            "Simplified: A={:?} B={:?}, {} edges",
            self.a,
            self.b,
            self.num_edges()
        );

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
        info!("{}", "BUILD & SIMPLIFY GRAPH".bold());
        self.drop_singletons();
        if get_flag("no_transform") {
            return vec![self.clone()];
        }
        // This does an initial sort of the B vertices.
        self.merge_twins();
        self.merge_adjacent_edges();
        {
            // For initial gluing we need to compute more data.
            // FIXME: Loop this a few times?
            let mut g = self.to_graph(get_flag("initial_glue"));
            let solution = initial_solution::initial_solution(&g);
            let to_glue = g.glue(&solution);
            self.self_crossings += self.glue_and_permute(solution, &to_glue, &g);
        }

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
        for l in self.connections_b.iter_mut() {
            l.sort_unstable();
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
        self.b = self.connections_b.len();
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
        self.b = self.connections_b.len();
        self.inverse = VecB::from(new_inv);
        let new_inv_len = self.inverse.iter().flatten().count();
        assert_eq!(inv_len, new_inv_len);
        self.self_crossings += self_crossings;
        info!(
            "Merged {} twins; {self_crossings} self crossings",
            Step::steps_between(&self.connections_b.len(), &self.b).unwrap()
        );
    }

    /// Given two adjacent nodes (u,v) in A.
    /// If the nbs of u and v are *only* connected to u and v, then we can merge u and v, and we can merge their neighbours.
    fn merge_adjacent_edges(&mut self) {
        let connections_a = self.reconstruct_a();
        self.sort_edges();

        let mut merged = 0;
        for ((x, cl), (y, cr)) in (NodeA(0)..self.a)
            .zip(connections_a.iter())
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
            // assert!(self[y].len() == self[*v].len());
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
        // self.reconstruct_a();
        self.drop_singletons();
    }

    /// Find pairs (u,v) with equal degree and neighbours(u) <= neighbours(v).
    fn dominating_pairs(&self, cr_range: &VecB<Range<NodeB>>) -> Before {
        let mut before = Before::from(vec![VecB::from(vec![Unordered; self.b.0]); self.b.0]);
        if get_flag("no_dominating_pairs") {
            return before;
        }

        let mut disjoint_pairs = 0;
        let mut dominating_pairs = 0;

        for u in NodeB(0)..self.b {
            let start = cr_range[u].start;
            let end = cr_range[u].end;
            for v in NodeB(0)..start {
                before[u][v] = After;
                disjoint_pairs += 1;
            }
            for v in end..self.b {
                before[u][v] = Before;
                disjoint_pairs += 1;
            }
            for v in start..end {
                if u == v || self[u] == self[v] {
                    continue;
                }
                // TODO: Skip this computation for pairs implied by the transitive closure.
                if self[u].last() < self[v].first() {
                    before[u][v] = Before;
                    disjoint_pairs += 1;
                    continue;
                }
                if self[u].first() > self[v].last() {
                    before[u][v] = After;
                    disjoint_pairs += 1;
                    continue;
                }

                // u[i] must be smaller than v[j] for all j>=floor(i*vl/ul)
                if is_dominating_pair(&self[u], &self[v]) {
                    before[u][v] = Before;
                    before[v][u] = After;
                    dominating_pairs += 1;
                }
            }
        }
        info!("Found {disjoint_pairs} disjoint pairs");
        info!("Found {dominating_pairs} dominating pairs");
        before
    }

    fn practical_dominating_pairs(&self, before: &mut Before, cr: &ReducedCrossings) {
        #[cfg(feature = "exact")]
        let do_pd = !get_flag("no_pd");
        #[cfg(not(feature = "exact"))]
        let do_pd = get_flag("pd");

        if !do_pd {
            info!("Found 0 practical dominating pairs (skipped)");
            return;
        }

        let mut practical_dominating_pairs = 0;
        let mut not_practical_dominating_pairs = 0;

        let cache = &mut KnapsackCache::default();

        let xs = (NodeB(0)..self.b).collect_vec();

        // For loop is reversed before is_pdp is more efficient with fixed v.
        for v in NodeB(0)..self.b {
            for u in NodeB(0)..v {
                let is_pdp = is_practically_dominating_pair(
                    u, v, before, &cr, &xs,
                    // TODO: Investigate if equality is OK
                    // Most of the time it is but it makes bugs :(
                    false, cache,
                );
                match is_pdp {
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

fn is_dominating_pair(u: &Vec<NodeA>, v: &Vec<NodeA>) -> bool {
    (0..u.len()).all(|i| {
        let j = (i * v.len()) / u.len();
        u[i] <= v[j]
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsPDP {
    // Equivalent to No but not counted.
    Skip,
    No,
    Yes,
}

/// Is u forced before v because there is no separating set?
/// Prefers v to be 'constant' and u to be the 'iterator'.
pub fn is_practically_dominating_pair(
    u: NodeB,
    v: NodeB,
    before: &Before,
    cr: &ReducedCrossings,
    xs: &[NodeB],
    allow_equality: bool,
    cache: &mut KnapsackCache,
) -> IsPDP {
    if u == v || before[v][u] != Unordered {
        return IsPDP::Skip;
    }

    // We want to prove that u < v.

    // Try to find a set X such that vXu is the strictly optimal rotation.
    // If no such set exists, then u < v.

    // TODO: Better handle this equality case. We have to be careful
    // to not introduce cycles.
    if cr[v][u] < 0 {
        return IsPDP::Skip;
    }

    // knapsack
    let mut target = P(cr[u][v] as i32, cr[u][v] as i32);
    if allow_equality {
        target -= P(1, 1);
    }

    // let strong_ks = !get_flag("no_strong_ks");

    // We do not consider x that must be before u and v, or after u and v.
    let points = xs.iter().filter_map(|&x| {
        if x == u || x == v {
            return None;
        }
        // if strong_ks {
        // x must be left of v.
        if before[v][x] == After {
            return None;
        }
        // x must be right of u.
        if before[u][x] == Before {
            return None;
        }
        // } else {
        //     // x must be left of u and v.
        //     if before[u][x] == After && before[v][x] == After {
        //         return None;
        //     }
        //     // x must be right of u and v.
        //     if before[u][x] == Before && before[v][x] == Before {
        //         return None;
        //     }
        // }

        // x that want to be between u and v (as in uxv) are not useful here.
        if cr[u][x] <= 0 && cr[v][x] >= 0 {
            return None;
        }
        Some(P(-cr[u][x] as i32, cr[v][x] as i32))
    });

    if !knapsack(target, points, false, cache) {
        IsPDP::Yes
    } else {
        IsPDP::No
    }
}

impl Graph {
    /// For each adjacent pair uv, as already ordered by the initial solution, test if we can glue them.
    /// That is the case if:
    /// - cr(u,v) <= 0, i.e. u wants to come before v.
    /// - u is (practically) dominated by v.
    /// - There is no separating set X such that vXu is optimal.
    /// Returns a list of indices i in the solution such that (i,i+1) should be glued.
    fn glue(&mut self, sol: &Solution) -> Vec<usize> {
        let mut pairs_to_glue = vec![];
        if !get_flag("initial_glue") {
            return pairs_to_glue;
        }
        for (i, (&u, &v)) in sol.iter().tuple_windows().enumerate() {
            if self.before[u][v] != Before {
                if !is_dominating_pair(&self[u], &self[v]) {
                    let is_pdp = is_practically_dominating_pair(
                        u,
                        v,
                        &self.before,
                        &self.reduced_crossings,
                        // TODO: Smaller slice
                        sol,
                        // NOTE: cr[u][v]=0 is allowed to still be PDP.
                        true,
                        &mut self.knapsack_cache,
                    );
                    if is_pdp != IsPDP::Yes {
                        info!("{i}: {u:?}<{v:?} cr[uv]={} before[uv]={:?} not dominating & not practically dominating: {is_pdp:?}  {:?}  {:?}", self.cr(u,v), self.before[u][v], self[u], self[v]);
                        continue;
                    } else {
                        info!("{i}: practically dominating.");
                    }
                } else {
                    info!("{i}: dominating.");
                }
            } else {
                info!("{i}: BEFORE");
            }
            if is_practically_glued_pair(
                u,
                v,
                &self.before,
                &self.reduced_crossings,
                // TODO: Smaller slice
                sol,
                // We require a blocking set to be a *strict* improvement.
                true,
                &mut self.knapsack_cache,
            ) == IsPDP::Yes
            {
                info!("{i}: GLUE!");
                pairs_to_glue.push(i);
            } else {
                info!("{i}: NO GLUE!");
            }
        }
        info!(
            "{}",
            format!("GLUING {} PAIRS", pairs_to_glue.len()).green()
        );
        pairs_to_glue
    }
}

/// Is v forced to directly follow u?
/// Assumes that u must be before v.
pub fn is_practically_glued_pair(
    u: NodeB,
    v: NodeB,
    before: &Before,
    cr: &ReducedCrossings,
    xs: &[NodeB],
    require_strict: bool,
    cache: &mut KnapsackCache,
) -> IsPDP {
    // // TODO: Better handle equality cases.
    // if u == v || before[u][v] != Before {
    //     return IsPDP::Skip;
    // }

    // if cr[u][v] >= 0 {
    //     return IsPDP::Skip;
    // }

    // We want to prove that there is no (non-empty?) X such that
    // cost(uXv) < cost(Xuv).
    // and
    // cost(uXv) < cost(uvX).
    // i.e.
    // cr(u, X) < 0 && cr(X, v) < 0

    let strong_ks = get_flag("strong_ks");

    // We do not consider x that must be before u and v, or after u and v.
    let points = xs.iter().filter_map(|&x| {
        if x == u || x == v {
            return None;
        }
        if strong_ks {
            // x must be left of u.
            if before[u][x] == After {
                return None;
            }
            // x must be right of v.
            if before[v][x] == Before {
                return None;
            }
        } else {
            // x must be left of u and v.
            if before[u][x] == After && before[v][x] == After {
                return None;
            }
            // x must be right of u and v.
            if before[u][x] == Before && before[v][x] == Before {
                return None;
            }
        }
        // x that want to be between v and u (as in vxu) are not useful here.
        if cr[u][x] >= 0 && -cr[v][x] >= 0 {
            return None;
        }
        Some(P(cr[u][x] as i32, -cr[v][x] as i32))
    });

    // FIXME TODO Do we have to allow equality here?
    let p = if require_strict { P(-1, -1) } else { P(0, 0) };
    if !knapsack(p, points, true, cache) {
        IsPDP::Yes
    } else {
        IsPDP::No
    }
}

impl GraphBuilder {
    /// When a cell is green and there is not a single red cell left-below it, fix the order of the pair.
    /// I.e.: When u < v in the current order, and for all (x,y) with x <= u < v <= y we want x < y, then fix u < v.
    fn boundary_pairs(&self, before: &mut Before) {
        if !get_flag("boundary_pairs") {
            // info!("Found 0 boundary pairs (skipped)");
            return;
        }

        let eq_boundary_pairs = get_flag("eq_boundary_pairs");

        let mut boundary_pairs = 0;

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
                    if let Some(pending) = pending {
                        if eq_boundary_pairs {
                            for u in pending..u {
                                before[u][v] = Before;
                                before[v][u] = After;
                                boundary_pairs += 1;
                            }
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
    pub fn reconstruct_a(&mut self) -> VecA<Vec<NodeB>> {
        self.a = self.connections_b.nb_len();
        self.b = self.connections_b.len();
        let mut connections_a: VecA<Vec<NodeB>> = VecA::new(self.a);
        for b in NodeB(0)..self.b {
            for &a in self.connections_b[b].iter() {
                connections_a[a].push(b);
            }
        }
        connections_a
    }

    /// Permute the nodes of B such that the given solution is simply 0..b.
    /// Furthermore, glue the pairs (i,i+1) as given by `to_glue`.
    /// Returns the score for newly created self-crossings.
    fn glue_and_permute(&mut self, solution: Solution, to_glue: &Vec<usize>, _g: &Graph) -> u64 {
        let mut score = 0;
        let mut target = 0;
        let mut last_i = usize::MAX / 2;
        for &i in to_glue {
            if i != last_i + 1 {
                target = i;
            }
            // Merge v into u.
            let u = solution[target];
            let v = solution[i + 1];

            info!("{i}: Merge {v} into {u}");

            let inv = std::mem::take(&mut self.inverse[v]);
            self.inverse[u].extend(inv);
            let cons = std::mem::take(&mut self.connections_b[v]);

            score += Self::edge_list_crossings(&self.connections_b[u], &cons);

            self.connections_b[u].extend(cons);

            last_i = i;
        }
        self.inverse = VecB::from(
            solution
                .iter()
                .filter_map(|&b| {
                    let x = std::mem::take(&mut self.inverse[b]);
                    if x.is_empty() {
                        None
                    } else {
                        Some(x)
                    }
                })
                .collect(),
        );
        self.connections_b = VecB::from(
            solution
                .iter()
                .filter_map(|&b| {
                    let x = std::mem::take(&mut self.connections_b[b]);
                    if x.is_empty() {
                        None
                    } else {
                        Some(x)
                    }
                })
                .collect(),
        );
        assert_eq!(self.inverse.len(), self.connections_b.len());
        self.b = self.connections_b.len();
        self.sort_edges();

        score
    }

    fn intervals(&self) -> VecB<Range<NodeA>> {
        VecB::from(
            self.connections_b
                .iter()
                .map(|b| {
                    b.first().copied().unwrap_or(NodeA(0))..b.last().copied().unwrap_or(NodeA(0))
                })
                .collect(),
        )
    }

    pub fn edge_list_crossings(e1: &Vec<NodeA>, e2: &Vec<NodeA>) -> u64 {
        // if e1.is_empty() || e2.is_empty() {
        //     return 0;
        // }
        // if e1.last().unwrap() <= e2.first().unwrap() {
        //     return 0;
        // }
        // if e1.first().unwrap() > e2.last().unwrap() {
        //     return CR::MAX as _;
        // }

        // For each in the left list, find how many in the right list are strictly smaller.
        let mut result: u64 = 0;
        let mut k = 0;
        for edge_i in e1 {
            while k < e2.len() && e2[k] < *edge_i {
                k += 1;
            }
            result += k as u64;
        }

        // Marginally faster but less robust.
        // for e1 in e1 {
        //     for e2 in e2 {
        //         if e2 < e1 {
        //             result += 1;
        //         }
        //     }
        // }

        result
    }

    pub fn one_node_crossings(&self, i: NodeB, j: NodeB) -> u64 {
        Self::edge_list_crossings(&self[i], &self[j])
    }

    fn crossings(
        &self,
    ) -> (
        u64,
        ReducedCrossings,
        VecB<Range<NodeB>>,
        VecB<NodeA>,
        VecB<NodeA>,
    ) {
        info!("CROSSINGS");
        let mut min_crossings = 0;
        let mut reduced_crossings = VecB::from(vec![VecB::new(self.b); self.b.0]);
        let mut cr_range = VecB::from(vec![self.b..NodeB(0); self.b.0]);
        // rightmost nb in each prefix.
        let prefix_max = VecB::from(
            self.connections_b
                .iter()
                .map(|x| x.last().copied().unwrap_or(NodeA(0)))
                .scan(NodeA(0), |rm, x| {
                    *rm = max(*rm, x);
                    Some(*rm)
                })
                .collect_vec(),
        );
        // leftmost nb in each suffix.
        let mut suffix_min = VecB::from(
            self.connections_b
                .iter()
                .rev()
                .map(|x| x.first().copied().unwrap_or(NodeA(usize::MAX)))
                .scan(NodeA(usize::MAX), |rm, x| {
                    *rm = min(*rm, x);
                    Some(*rm)
                })
                .collect_vec(),
        );
        suffix_min.reverse();

        let try_into = |cr: i64, _i, _j| {
            cr as CR
            // cr.try_into().unwrap_or_else(|_| {
            //     panic!("Crossings between {i} and {j} is too large for type: {cr}",)
            // })
        };

        for i in NodeB(0)..self.b {
            if self[i].is_empty() {
                continue;
            }
            // Only consider the range of possibly intersecting nodes.
            let li = *self[i].first().unwrap();
            let ri = *self[i].last().unwrap();

            let jl = NodeB(
                prefix_max
                    .binary_search_by(|x| {
                        if x < &li {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                    .unwrap_or_else(|x| x),
            );

            let jr = NodeB(
                suffix_min
                    .binary_search_by(|x| {
                        if x <= &ri {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                    .unwrap_or_else(|x| x),
            );

            if !get_flag("no_lazy_cr") {
                // Instead of computing the true value, we just put a large value, since i will never come before them anyway.
                reduced_crossings[i].v[..jl.0].fill(CR::MAX / 2048);
                // We do +1 so that `-cr[u][v]` fits in the CR type as well.
                reduced_crossings[i].v[jr.0..].fill((CR::MIN + 1) / 2048);
            } else {
                for j in NodeB(0)..jl {
                    reduced_crossings[i].v[j.0] = (self[i].len() * self[j].len()) as _;
                }
                for j in jr..self.b {
                    reduced_crossings[i].v[j.0] = -((self[i].len() * self[j].len()) as CR);
                }
            }

            cr_range[i] = jl..jr;

            for j in jl..jr {
                // HOT: 20% of time is spent on the inefficient memory access here.
                let cij = self.one_node_crossings(i, j);
                let cji = self.one_node_crossings(j, i);
                if i < j {
                    min_crossings += min(cij, cji);
                }
                let cr = cij as i64 - cji as i64;
                reduced_crossings[i][j] = try_into(cr, i, j);
            }
        }
        (
            min_crossings,
            reduced_crossings,
            cr_range,
            prefix_max,
            suffix_min,
        )
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

// impl Index<NodeA> for GraphBuilder {
//     type Output = Vec<NodeB>;

//     fn index(&self, index: NodeA) -> &Self::Output {
//         &self.connections_a[index]
//     }
// }

// impl IndexMut<NodeA> for GraphBuilder {
//     fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
//         &mut self.connections_a[index]
//     }
// }

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
