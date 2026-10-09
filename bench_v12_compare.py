"""Compare v1.3 (current) spike counts against v1.2 captured baseline."""
import sys
import numpy as np

sys.path.insert(0, ".")

for name in ("toy", "malecns"):
    a = np.load(f"bench_v12/{name}_a.npy")
    b = np.load(f"bench_v12/{name}_b.npy")
    eq = bool(np.array_equal(a, b))
    diff = int(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())
    print(f"{name}: shape={a.shape}, sum_v12={int(a.sum())}, sum_now={int(b.sum())}, "
          f"bit_identical={eq}, total_abs_diff={diff}")
    if not eq:
        nz = np.nonzero(a - b)[0]
        print(f"  first 5 differing: {nz[:5].tolist()}")
        print(f"  v1.2 values: {a[nz[:5]].tolist()}")
        print(f"  v1.3 values: {b[nz[:5]].tolist()}")
