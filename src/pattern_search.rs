use itertools::Itertools;

use crate::{
    get_flag,
    graph::{draw, Graph, Solution},
};

pub fn pattern_search(g: &Graph, sol: &Solution) {
    if !get_flag("pattern_search") {
        return;
    }

    // Look for:
    // vXu
    // with:
    // cuv < cvu (we want u before v)
    // for each x in X:
    // vxu is not optimal, i.e. uvx or xuv is better.
    //   i.e. u can be swapped with vx or v can be swapped with xu.
    // vXu is optimal, i.e. uvX or Xiv is worse. (This is implied by the solution.)

    // vxu < uvx && vxu < xuv
    let x_in_middle_is_optimal = |v, x, u| {
        let vxu = g.c(v, x) + g.c(x, u) + g.c(v, u);
        let uvx = g.c(u, v) + g.c(v, x) + g.c(u, x);
        let xuv = g.c(x, u) + g.c(u, v) + g.c(x, v);
        vxu <= uvx && vxu <= xuv
    };

    // Loop over v..u at positions i < j.
    for (j, &u) in sol.iter().enumerate() {
        'uv: for (i, &v) in sol[..j].iter().enumerate() {
            if v == u {
                break;
            }
            // If v should anyway be before u, skip.
            if g.c(v, u) <= g.c(u, v) {
                continue;
            }

            // Check that vxu is not optimal for all x.
            for &x in &sol[i + 1..j] {
                if x_in_middle_is_optimal(v, x, u) {
                    continue 'uv;
                }
            }

            // // Check that vxyu is not optimal.
            // for (&x, &y) in sol[i + 1..j].iter().tuple_combinations() {
            //     // Try swapping u with vxy.
            //     let swap_u_better = g.node_score(u, v) + g.node_score(u, x)
            //         < g.node_score(v, u) + g.node_score(x, u);
            //     // Try swapping v with xyu.
            //     let swap_v_better = g.node_score(x, v) + g.node_score(u, v)
            //         < g.node_score(v, x) + g.node_score(v, u);

            //     if !swap_u_better && !swap_v_better {
            //         continue 'uv;
            //     }
            // }

            // vxu is not optimal, but vXu is.
            eprintln!("{i} {j}");
            for &x in &sol[i..=j] {
                eprintln!("{x} => {:?}", g[x]);
            }
            eprintln!();

            let connections = sol[i..=j].iter().map(|&x| g[x].clone()).collect_vec();
            draw(&connections, false);
        }
    }
}
