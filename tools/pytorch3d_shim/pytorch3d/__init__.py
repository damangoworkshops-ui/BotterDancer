# BotterDancer vendored shim (2026-07-10): minimal pure-Python subset of pytorch3d,
# containing ONLY pytorch3d.transforms (rotation math, verbatim from the official
# facebookresearch/pytorch3d repo, BSD license) + a stub pytorch3d.ops.knn.
# NO compiled CUDA extension (renderer/structures) -- GVHMR's own Renderer class
# already degrades gracefully (see hmr4d/utils/vis/renderer.py optional-import patch)
# when those are absent, which is fine since BotterDancer skips GVHMR's debug rendering.
__version__ = "0.0.0-botterdancer-shim"
