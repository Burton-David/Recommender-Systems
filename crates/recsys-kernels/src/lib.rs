//! Hot-path kernels for the recommender_systems library.
//!
//! Exposes a small surface — one function per algorithm we've measured into the
//! ground via the Phase 1.1 baseline. Everything else stays in Python.

use ndarray::{ArrayViewMut1, ArrayViewMut2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Run `epochs` full passes of BPR SGD against the existing factor matrices.
///
/// The Python caller owns `user_factors` and `item_factors` and passes them in
/// as writable views; we mutate them in place. `positives` is a flattened
/// `(n_pos, 2)` int64 array where each row is `[user_idx, item_idx]`, and
/// `observed_flat` is the row-major flattening of the `(n_users, n_items)` bool
/// matrix that tells us which items each user has already interacted with — used
/// to reject negatives that aren't actually negatives.
///
/// Mirrors `BPR._step` in `bpr.py`, with the per-positive negative-resample
/// loop and the same sigmoid-margin gradient updates. Random numbers come from
/// PCG64 (which numpy also uses), seeded by `seed`; results are deterministic
/// for a given seed but will not bit-match the pure-Python BPR because the two
/// implementations consume RNG bytes in different orders. Equivalence is
/// established at the recommendation-quality level via a regression test.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bpr_train(
    mut user_factors: PyReadwriteArray2<f64>,
    mut item_factors: PyReadwriteArray2<f64>,
    positives: PyReadonlyArray2<i64>,
    observed_flat: PyReadonlyArray1<bool>,
    n_items: usize,
    epochs: usize,
    learning_rate: f64,
    reg: f64,
    seed: u64,
) -> PyResult<()> {
    let mut user_factors = user_factors.as_array_mut();
    let mut item_factors = item_factors.as_array_mut();
    let positives = positives.as_array();
    let observed = observed_flat.as_array();

    let n_pos = positives.nrows();
    let n_items_i64 = n_items as i64;
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n_pos).collect();

    for _ in 0..epochs {
        shuffle(&mut order, &mut rng);
        for &idx in &order {
            let u = positives[[idx, 0]] as usize;
            let i = positives[[idx, 1]] as usize;
            let mut j = rng.gen_range(0..n_items_i64) as usize;
            while observed[u * n_items + j] {
                j = rng.gen_range(0..n_items_i64) as usize;
            }
            step(&mut user_factors, &mut item_factors, u, i, j, learning_rate, reg);
        }
    }
    Ok(())
}

/// One BPR SGD update — splits user_factors and item_factors into the three
/// disjoint rows we need, computes the sigmoid-margin gradient, and applies
/// the update in place.
#[inline]
fn step(
    user_factors: &mut ArrayViewMut2<f64>,
    item_factors: &mut ArrayViewMut2<f64>,
    u: usize,
    i: usize,
    j: usize,
    lr: f64,
    reg: f64,
) {
    // Snapshot the rows we read so the in-place updates below don't trip the
    // borrow checker on overlapping mutable views into item_factors.
    let u_vec: Vec<f64> = user_factors.row(u).to_vec();
    let i_vec: Vec<f64> = item_factors.row(i).to_vec();
    let j_vec: Vec<f64> = item_factors.row(j).to_vec();
    let n = u_vec.len();

    let mut margin = 0.0;
    for k in 0..n {
        margin += u_vec[k] * (i_vec[k] - j_vec[k]);
    }
    let sig = 1.0 / (1.0 + margin.exp()); // = sigmoid(-margin); saturates safely at the tails

    apply_update(&mut user_factors.row_mut(u), &u_vec, &i_vec, &j_vec, sig, lr, reg, UpdateKind::User);
    apply_update(&mut item_factors.row_mut(i), &i_vec, &u_vec, &u_vec, sig, lr, reg, UpdateKind::PosItem);
    apply_update(&mut item_factors.row_mut(j), &j_vec, &u_vec, &u_vec, sig, lr, reg, UpdateKind::NegItem);
}

enum UpdateKind {
    User,
    PosItem,
    NegItem,
}

#[inline]
fn apply_update(
    target: &mut ArrayViewMut1<f64>,
    own: &[f64],
    a: &[f64],
    b: &[f64],
    sig: f64,
    lr: f64,
    reg: f64,
    kind: UpdateKind,
) {
    let n = own.len();
    for k in 0..n {
        let grad = match kind {
            UpdateKind::User => sig * (a[k] - b[k]),
            UpdateKind::PosItem => sig * a[k],
            UpdateKind::NegItem => -sig * a[k],
        };
        target[k] = own[k] + lr * (grad - reg * own[k]);
    }
}

#[inline]
fn shuffle<T>(slice: &mut [T], rng: &mut impl Rng) {
    // Fisher-Yates so the per-epoch positive ordering matches what numpy's
    // permutation gives semantically — independent draws, full coverage.
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0..=i);
        slice.swap(i, j);
    }
}

#[pymodule]
fn _kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bpr_train, m)?)?;
    Ok(())
}
