# TODO: speed up the loop-closure pose graph in the chunked refine

Priority: **high**. Status: found 2026-10-23 on kladno_dira (5681 images, BA mode auto → chunked); **done 2026-10-23**
(see Result).

## Symptom

`chunked.pose_graph` (called from `pipeline._refine_chunks`):
- with the chunk edges only (26,638 edges, 5010 priors): 58 iterations, **60 s**;
- with the ~4000 loop-closure edges of a revisit pass (`loops=`, Huber): **~37 min**, on one CPU core (the GUI
  process at 100 %), with no log line until it ends. Revisit pass 1 logged at 10:28 UTC, the next triangulation
  started at 11:05; after pass 2 (11:09) it ran ~35 min, after pass 3 (11:48) **55 min** (11:48 → 12:43 UTC): ~2 h of the refine in total. A triangulation of all 5681 images takes only ~4 min.
- Its result is stored only in `stats["revisits"][k]["pose_graph"]` (report.json), not logged.

## Likely causes (to measure first)

- Fill-in: loop edges join cameras far apart in capture order, so the normal equations lose their banded structure
  (`SPARSE_NORMAL_CHOLESKY`; Ceres' default backend here is SUITE_SPARSE / CHOLMOD, checked in the container).
- Iteration count: up to `max_iterations=200`; with Cauchy chunk edges + Huber loops it may run to the cap
  with tiny steps (no settle rule here, unlike `refine.prior_ba`).
- Single-threaded factorization (CHOLMOD without a threaded BLAS; `num_threads` only parallelizes the Jacobian).

## Plan

1. Log it: "pose graph with N loops: k it (termination), s" after each revisit pass, and a GUI state line detail
   (Ceres iteration callback: "pose graph iteration k / 200").
2. Reproduce offline on a saved problem (dump `init`, `edges`, `loops`, priors from a kladno_dira run to
   `build/`; dira2k for a smaller case) and time: iterations and seconds per iteration, CHOLMOD vs EIGEN_SPARSE,
   ordering (AMD vs nested dissection `linear_solver_ordering_type`), `ITERATIVE_SCHUR` / `CGNR` with a Jacobi or
   SUBSET preconditioner, `function_tolerance` 1e-6 → 1e-4, a settle rule on camera motion as in `refine.Settle`.
3. Pick the fastest that gives the same split after the pass (within ~1 cm) on dira2k and kladno_dira.

Do not test on the full kladno_dira refine (hours); the dumped problem alone is enough.

## Result (2026-10-23)

- **Cause 1, the BLAS:** Ceres' CHOLMOD ran on Ubuntu's reference netlib BLAS (single-threaded, unoptimized), and the
  loop edges make the Cholesky factor dense. With OpenBLAS (`libopenblas0-pthread`, a late Dockerfile layer;
  update-alternatives picks it) a synthetic 2000-camera graph with 1500 loops took 4.0 s instead of 93.5 s, with the
  same iterations and solution. This also affects every other CHOLMOD / LAPACK user (COLMAP's BA through Ceres).
- **Cause 2, the solver:** even with OpenBLAS the factorization dominates. CGNR (conjugate gradients on the normal
  equations, Jacobi preconditioner) needs no factorization. On the real dira2k pass-1 problem (2000 cameras, 8723 chunk
  edges, 1412 loops; dumped with `MODELFORGE_DUMP_POSEGRAPH=dir`, solved by `build/pg_bench.py DUMP VARIANTS`):

  | solver | iterations | time | final cost |
  |---|---|---|---|
  | Cholesky, reference BLAS (the earlier run) | 54 | 368 s | |
  | Cholesky, OpenBLAS | 54 | 22.5 s | 37476 |
  | **CGNR** | 59 | **1.1 s** | 37413 |
  | dogleg (Cholesky) | 54 | 19.1 s | 37476 |
  | function_tolerance 1e-4 | 26 | 7.7 s | 37498 |
  | EIGEN_SPARSE (synthetic 5681 cameras) | 8 | 2258 s | |

  CGNR ends in a slightly different minimum (up to 4 m off at a few cameras, lower cost), so it was judged by what
  follows: two whole chunked refines of dira2k (`build/dira2k_pg`, `build/dira2k_cgnr`, `build/refine_solver.py`):
  revisit split per pass 2.51 / 0.385 / 0.340 m (Cholesky) vs 2.51 / 0.382 / 0.353 m (CGNR), after the global passes
  0.085 vs 0.080 m; like-for-like (`build/split_check.py`, pairs > 60 s) 0.05 / 0.21 m vs 0.06 / 0.22 m, both 80 %
  merged; final 0.590 vs 0.589 px. Loop pose graphs 48 + 55 + 65 s (Cholesky, OpenBLAS) vs **3.4 + 2.9 + 1.1 s**; the
  chunk-only graph 10.6 → 2.1 s.
- **Now:** `Chunked.posegraph_solver = "cgnr"` (default; "cholesky" stays available), `pose_graph(solver=,
  on_iteration=, token=)`: each loop pose graph is logged ("revisit pass k: pose graph with N loop edges: i it, s"),
  the GUI shows "pose graph …: iteration k / 200", Stop aborts it.
- Not re-run on kladno_dira 5681 (5 h); expected: the ~2 h of loop pose graphs become well under a minute.
