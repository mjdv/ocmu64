# PACE 2024 - One sided crossing minimization - OCMU64

This is a solution to the 2024 [PACE challenge](https://pacechallenge.org/), by
@mjdv and @RagnarGrootKoerkamp.

In particular, we solve the _one sided crossing minimization_ problem:
- Given is a bipartite graph $(A, B)$, that is drawn in the plane with straight
  lines, with vertices in $A$ at $(0, i)$ and vertices in $B$ at $(1, j)$.
- The order of $A$ is fixed. The goal is to find an order of $B$ that minimizes
  the number of crossings.
  
This is an **exact solver**.
- On the exact track, it solves around 75% of cases (given half an hour per case).
- On the parameterized tack, is solves all 100 cases on optil.io in around 3 seconds total.

## Algorithm overview
Our method uses branch and bound.
- First, we find an initial solution using some simple local search (swapping
  adjacent elements and optimally inserting elements elsewhere).
- The branch and bound works from left to right. At each step, we try (a subset
  of) all remaining nodes as the next node in the order.
    - We track the score of the _prefix_ $P$ fixed so far, and the number of crossings
    between $P$ and the remaining _tail_ $T$.
    - We also use the 'trivial' lower bound for the number of crossings in the tail.
    - We prune branches where the score is higher than the best score seen so far.

Some of the more interesting optimizations:
- **Graph simplification**: We merge identical edges, drop empty nodes from both
  components, and sort the nodes of $B$ by their position in the initial
  solution, for more efficient memory accesses. We also split 'disjoint' components.
- **Caching**: For each B&B state we cache (among other things) the best lower bound
  found so far, or the score of the best solution if known.
  - If a tail has not been seen yet, use the bound for the longest suffix of the
    tail that has been seen.
- **Optimal insert**: In the B&B, at each step we don't just append the next
  node, but insert it in the optimal position in the prefix.
- **Dominating pairs**: For some pairs of nodes $(u,v)$ in $B$, we can prove
  that $u$ must always come before $v$. We pre-compute these pairs, and never
  try $v$ as the next node when $u$ hasn't been chosen yet.
- **Practical dominating pairs**: A stronger variant of the above that not only
  considers $u$ and $v$ 'in isolation', but in context of the other nodes in
  $B$. Sometimes $v$ could 'in theory' come before $u$, but we can prove that in
  practice this can never be optimal (or at least, not strictly better than
  having $u$ before $v$).
- **Local dominating pairs**: Same as practical dominating pairs above, but recomputed in every new B&B state.
- **On-the-fly gluing**: If there is a vertex $u$ in the tail such that all
  other $v$ in the tail are better (not worse) after $u$, then just _glue_ $u$ onto the prefix.
There are some more optimizations that are disabled by default. See [notes.md](notes.md).

Furthermore we do a number of algorithmic optimizations to implement everything
efficiently. Mostly we ensure that the tail always remains sorted, so that loops
over it can often be reduced to only a small subset of it.

## Usage instructions
This solver is written in Rust.
- First, install Rust, e.g. using [rustup](https://rustup.rs/).
- Then, install Rust `nightly`, using `rustup install nightly`, followed by
  `rustup default nightly`.
- Then build the binary for the '`submit`' profile.
  - For the parameterized track (the default):
    `RUSTFLAGS="-Ctarget-cpu=native" cargo build --profile submit`
  - For the exact track:
    `RUSTFLAGS="-Ctarget-cpu=native" cargo build --profile submit --features exact`
- Find the binary at `target/submit/ocmu64`.
- Run as `ocmu64 < input > output`.
