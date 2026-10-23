# TODO: speed up the loop-closure pose graph in the chunked refine

Priority: **high**. Status: found 2026-10-23 on kladno_dira (5681 images, BA mode auto → chunked), not started.

## Symptom

`chunked.pose_graph` (called from `pipeline._refine_chunks`):
- with the chunk edges only (26,638 edges, 5010 priors): 58 iterations, **60 s**;
- with the ~4000 loop-closure edges of a revisit pass (`loops=`, Huber): **~37 min**, on one CPU core (the GUI
  process at 100 %), with no log line until it ends. Revisit pass 1 logged at 10:28 UTC, the next triangulation
  started at 11:05; after pass 2 (11:09) it ran again as long. A triangulation of all 5681 images takes only ~4 min,
  so with 3 revisit passes this solve is ~2 h of the refine.
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
