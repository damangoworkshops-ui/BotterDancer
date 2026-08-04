# BotterDancer shim: pytorch3d.ops.knn is only used by hmr4d's BEDLAM dataset loader
# (training-time only, never on the demo.py extraction path). Stub that fails loudly
# and specifically if anyone ever calls it, instead of silently returning wrong data.
def knn(*args, **kwargs):
    raise NotImplementedError(
        "pytorch3d.ops.knn is not available in this minimal shim (compiled CUDA op). "
        "This should only be reachable from GVHMR training/dataset code, not the "
        "demo.py extraction path BotterDancer uses."
    )
