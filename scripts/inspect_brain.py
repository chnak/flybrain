"""Inspect flybrainer Brain / KCMBONPlasticity / PlasticityConfig for service design."""
import inspect

import flybrainer

print("=" * 70)
print("Brain.__doc__")
print("=" * 70)
print(flybrainer.Brain.__doc__ or "(no doc)")

print()
print("=" * 70)
print("Brain public methods (instance)")
print("=" * 70)
for name, member in inspect.getmembers(flybrainer.Brain, predicate=inspect.isfunction):
    if not name.startswith("_"):
        print(f"  - {name}{inspect.signature(member)}")

print()
print("=" * 70)
print("Brain class-level / constructor")
print("=" * 70)
print(f"  - __init__{inspect.signature(flybrainer.Brain.__init__)}")
for name, member in inspect.getmembers(flybrainer.Brain):
    if name.startswith("_") and not name.startswith("__"):
        print(f"  - (private) {name}")

print()
print("=" * 70)
print("PlasticityConfig")
print("=" * 70)
print(f"  - __init__{inspect.signature(flybrainer.PlasticityConfig.__init__)}")
print(f"  - fields: {list(flybrainer.PlasticityConfig.__dataclass_fields__.keys()) if hasattr(flybrainer.PlasticityConfig, '__dataclass_fields__') else 'n/a'}")
print(flybrainer.PlasticityConfig.__doc__ or "(no doc)")

print()
print("=" * 70)
print("KCMBONPlasticity")
print("=" * 70)
for name, member in inspect.getmembers(flybrainer.KCMBONPlasticity, predicate=inspect.isfunction):
    if not name.startswith("_"):
        print(f"  - {name}{inspect.signature(member)}")

print()
print("=" * 70)
print("Stimulus / ObservationResult / Decision / Prediction")
print("=" * 70)
for cls_name in ["Stimulus", "ObservationResult", "Decision", "Prediction", "SensorFrame"]:
    cls = getattr(flybrainer, cls_name, None)
    if cls is None:
        continue
    print(f"\n--- {cls_name} ---")
    print(f"  - __init__{inspect.signature(cls.__init__)}")
    if hasattr(cls, "__dataclass_fields__"):
        for f, meta in cls.__dataclass_fields__.items():
            print(f"    field: {f}: {meta.type}")
    print(f"  doc: {(cls.__doc__ or '(no doc)').strip()[:200]}")

print()
print("=" * 70)
print("Required populations")
print("=" * 70)
print(flybrainer.REQUIRED_POPULATIONS)