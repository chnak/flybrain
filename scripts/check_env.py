"""Check environment: CPU, MaleCNS data, previous tests."""
import os
from pathlib import Path

print("=== CPU ===")
print("logical cores:", os.cpu_count())
try:
    import psutil
    print("physical cores:", psutil.cpu_count(logical=False))
    print("affinity:", psutil.Process().cpu_affinity())
except ImportError:
    print("(psutil not installed)")

print("\n=== flybrain data ===")
from flybrainer.paths import PATHS
print("root:", PATHS.root, "exists:", PATHS.root.exists())
print("graph.npz:", PATHS.graph, "exists:", PATHS.graph.exists(), "size MB:", PATHS.graph.stat().st_size / 1e6 if PATHS.graph.exists() else 0)
print("malecns:", PATHS.malecns, "exists:", PATHS.malecns.exists())

fb = Path.home() / ".flybrain"
for sub in ("graphs", "logs", "sessions", "toy_graph.npz"):
    p = fb / sub
    if p.exists():
        if p.is_dir():
            files = list(p.iterdir())
            print(f"  {sub}/: {len(files)} entries")
            for f in files[:3]:
                print(f"    {f.name}")
        else:
            print(f"  {sub}: {p.stat().st_size / 1e6:.2f} MB")
