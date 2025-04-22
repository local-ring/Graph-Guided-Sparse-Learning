# driver_multiprocess.py
from __future__ import annotations
import os, multiprocessing, concurrent.futures, numpy as np
from collections import defaultdict
from tqdm import tqdm

from random_ensemble import RandomEnsemble

# -------------------------------------------------------------------
def _run_single_n(n: int,
                  d: int = 1000, k: int = 50,
                  models=("GFL_Matlab", "GFL_normalized"),
                  num_replications: int = 10,
                  seed: int | None = 42) -> tuple[int, dict]:
    """Worker function – must be importable."""
    if seed is not None:
        np.random.seed(seed + n)

    re = RandomEnsemble(n=n, d=d, k=k,
                        h_total=2, h_selected=1, h_rest=1,
                        gamma=0.5, p=0.7, q=0.2,
                        models=list(models),
                        num_replications=num_replications)
    return n, re.main(show_progress=True)

# -------------------------------------------------------------------
def main():
    sample_sizes = np.arange(50, 600, 100)           # 50,150,…,550
    # max_workers  = min(len(sample_sizes), os.cpu_count(), 2)
    max_workers = 1

    mp_ctx = multiprocessing.get_context("spawn")
    tqdm.set_lock(mp_ctx.RLock())                     # shared lock

    outer_bar = tqdm(total=len(sample_sizes), desc="Sample‑size grid",
                     position=0)

    agg = defaultdict(lambda: defaultdict(list))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),)) as pool:

        fut_to_n = {pool.submit(_run_single_n, n): n for n in sample_sizes}

        for fut in concurrent.futures.as_completed(fut_to_n):
            n, acc_dict = fut.result()
            for model, acc in acc_dict.items():
                agg[n][model] = acc
            outer_bar.update(1)

    outer_bar.close()

    # ------------ print summary ------------------------------------
    print("\n=== Support‑recovery accuracy ===")
    for n in sorted(agg):
        for model, acc in agg[n].items():
            print(f"n={n:3d} | {model:14s} : "
                  f"mean={np.mean(acc):.4f}  std={np.std(acc):.4f}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()   # safe cross‑platform
    main()
