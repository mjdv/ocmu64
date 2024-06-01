$$
\renewcommand{\p}{\prec}
\renewcommand{\pe}{\preceq}
\renewcommand{\g}{\cdot}
$$


# Problem statement

Given is a bipartite graph $A\cup B$, where the order of $A$ is fixed.
Permute $B$ such that the number of crossings when drawn in the plane is minimized.


# Notation

-   $[n] = \{0, \dots, n-1\}$.
-   $u,v\in B$ are vertices of $B$.
-   Vertices are also identified by their neighbours in $A$, $U\sim \Gamma(u)$,
    where we assume $U$ is sorted by the order of $A$.
-   We write $u\p v$ when $u$ comes before $v$ in $B$.
-   $c(u,v)$ is the number of crossings when $u\p v$.
-   $r(u,v) = c(u,v)-c(v,u)$ is the *reduced* number of crossings.
-   $c(u,v,w) = c(u,v) + c(v,w) + c(u,w)$ is the number of crossings when $u\p v\p w$.
-   When we write $Z\subseteq A$, this is taken to mean a weighted/multiset of
    vertices of $A$.
-   When we write $Z\subseteq B$, we mean that $Z$ is a subset of vertices of $B$
    and identified with the (weighted) union of their neighbours in $A$.


# Overview

At a high level, we implement branch and bound (B&B) on the order of $B$.

-   We try each vertex as *the next* one and fix it.
-   Compute a lower bound on the cost of the tail using the sum over pairwise $\min(c(u,v),c(v,u))$.
-   Do *optimal insert*: Each time a next vertex is fixed, it is inserted in the optimal
    position in the prefix $P$.
-   Cache the optimal cost of each tail $T$ to reduce double work.


# Ordering lemmas

Here we derive conditions on when we can force $u\pe v$, thereby reducing the
search space of possible optimal solutions.

**Lemma O1.** (Weak order)
When $\max U \leq \min V$, $u\pe v$.

**Lemma O2.** (Stronger order)
Suppose $|U| = |V|$ and $U_i \leq V_i$. Then $u\pe v$.

**Lemma O3.** (Strong order)
Suppose $|U| = n$, $|V| = m$, and for all $i\in [n]$:
$$
U_i \leq V_{\lfloor i\cdot m/n \rfloor},
$$
then $u\pe v$.
Note that the condition is equivalent to the symmetric version: for all $j\in [m]$:
$$
U_{\lceil j\cdot n/m \rceil} \leq V_j.
$$

Note, Lemma 3 implies Lemmas 1 and 2.

**Lemma O4.** (Optimal order)
In an optimal ordering, for any consecutive $u\p z\p v$ (where each can be either a single
vertex or the union of an interval of vertices), it most hold that:

\begin{align}\label{eq:opt3}
c(u,z,v) &\leq c(z,u,v),&
c(u,z,v) &\leq c(v,z,u).\\
\end{align}

**Lemma O5.** (Strong order is optimal)
When the condition in Lemma 3 does not hold, there exists a $Z\subseteq A$ such
that $c(Z,v,u) < c(u,Z,v)$ or $c(v,u,Z) < c(u,Z,v)$.

**Lemma O6.** (Practical order)
When for every $Z\subseteq B$ we have $c(u,Z,v) \leq c(z,u,v)$ and $c(u,Z,v) \leq c(v,z,u)$,
then $u\pe v$.
A set $Z\subseteq B$ that contradicts one of these inequalities is called a *blocking* set.

**Algorithm O7.** (Practical order algorithm) The condition of Lemma 6 can be checked
using a Knapsack-like method. (TODO)

**Corollary O8.** (Practical tail order)
When some prefix $P$ of vertices of $B$ is already fixed and $u$ and $v$ are in the
tail, and the condition of Lemma 6 is satisfied for all subsets $Z\subseteq T$
of the tail $T = B-P$, then $u\pe v$ in any optimal ordering of $T$.

**Algorithm O9.** (Practical tail order algorithm)
At the very start, and every time the B&B fixes a next vertex of the prefix, the condition of Lemma 6
is checked (using Algorithm 7) for all remaining pairs in the tail.

TODO: This should be made more efficient. Either do not check always, or check
incrementally, or cache these results. E.g. for each $(u,v)$ for which we do not
yet have $u\pe v$, we could store an example of a blocking set $Z$, and only
redo the computation when a vertex of $Z$ is fixed (so that it is not available
anymore to be used in between $u$ and $v$).

**Lemma O10.**
When $u\pe v$, this implies $c(u,v) \leq c(v,u)$, or equivalently, $r(u,v) \leq 0$.


# Gluing lemmas

We write $u\g v$ when $u$ and $v$ are glued together (in this order) in all (or
at least one?) optimal orderings.

When $u\g v$ and $u$ is fixed (optimally inserted into the prefix $P$), we
immediate fix $v$ (optimally insert it into $P$).

**Lemma G1.** (Weak gluing)
When $u=v$, $u\g v$.
(In this case the order doesn't matter.)

**Remark G2.** (No strong gluing???)
When $u\neq v$, one can always find a
$Z\subseteq A$ such that $c(u,Z,v) < c(u,v,Z)$ and $c(u,Z,v) < c(Z,u,v)$.

(This doesn't seem 100% true, but it seems hard to guarantee
gluing when arbitrary $Z\subseteq A$ are allowed.)

**Lemma G3.** (Practical gluing)
When $u\pe v$ (because of one of the ordering lemmas), and there is no
$Z\subseteq A$ such that $c(u,Z,v) < c(u,v,Z)$ and $c(u,Z,v) < c(Z,u,v)$,
then $u\g v$.

**Algorithm G4.** (Practical gluing algorithm)
At the very start, and every time the B&B fixes a next vertex of the prefix, the
condition of Lemma G3 is checked for all remaining pairs in the tail.


# Possible future lemmas

**Lemma X1.**
Suppose the tail $T$ is a local minimum, i.e., swaps (and maybe rotations?) do
not improve it.
We try to fix each $u\in T$ as the next vertex. Can we prove that the cost of
moving $u$ to the front increases as we iterate over all $u$?
