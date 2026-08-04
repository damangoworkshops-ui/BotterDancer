# BotterDancer shim: ECHTES Submodul, damit `import pytorch3d.ops.knn as knn`
# (hmr4d geo_transform.py) bindet. Vorher gab es nur die Funktion in ops/__init__.py:
# der Import warf ModuleNotFoundError, der Optional-Patcher machte daraus knn=None,
# und ein Trainingspfad-Aufruf ergab einen kryptischen AttributeError auf None
# statt dieser klaren Meldung (Review 2026-07-19).
_MSG = (
    "pytorch3d.ops.knn is not available in this minimal shim (compiled CUDA op). "
    "This should only be reachable from GVHMR training/dataset code, not the "
    "demo.py extraction path BotterDancer uses."
)


def knn_points(*args, **kwargs):
    raise NotImplementedError(_MSG)


def knn_gather(*args, **kwargs):
    raise NotImplementedError(_MSG)
