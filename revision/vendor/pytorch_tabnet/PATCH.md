# Vendored pytorch-tabnet 4.1.0 (T2.1, 2026-10-19)

Source: `pytorch-tabnet==4.1.0` wheel from PyPI (last release July 2023, unmaintained),
installed with `pip install --no-deps` into site-packages and copied here file by file
(`pytorch_tabnet/*.py` plus the MIT `LICENSE` from the dist-info). site-packages was NOT
modified. The revision scripts put `revision/vendor` at the front of `sys.path` before
`import pytorch_tabnet`, and `extra_models.py` asserts at run time that the imported module
comes from this directory (recorded in `t21_run_meta.json`).

## Why

Under `python -W error` (the revision convention for every run) the package does not import
in the frozen environment (scipy 1.17.1):

```
File ".../pytorch_tabnet/multiclass_utils.py", line 13
    from scipy.sparse.base import spmatrix
DeprecationWarning: Please import `spmatrix` from the `scipy.sparse` namespace; the
`scipy.sparse.base` namespace is deprecated and will be removed in SciPy 2.0.0.
```

`scipy.sparse.base` is a deprecated private alias of `scipy.sparse`; `spmatrix` is the same
object in both namespaces, so the patch changes nothing in behaviour.

## Exact diff (the only change)

```
--- site-packages/pytorch_tabnet/multiclass_utils.py
+++ revision/vendor/pytorch_tabnet/multiclass_utils.py
@@ -10,7 +10,7 @@
 from itertools import chain

 from scipy.sparse import issparse
-from scipy.sparse.base import spmatrix
+from scipy.sparse import spmatrix  # T2.1 vendor patch: scipy.sparse.base is deprecated (DeprecationWarning under -W error)
 from scipy.sparse import dok_matrix
 from scipy.sparse import lil_matrix
 import scipy.sparse as sp
```

All other files are byte-identical to the installed 4.1.0 wheel (checked with `cmp` at vendoring time).

## Warnings that remain and are excluded BY NAME (not patched, not suppressed globally)

Both are informational `UserWarning`s raised unconditionally by the library's training
callbacks (`callbacks.py`, `abstract_model.py`); they carry no diagnostic content for this
use, so `extra_models.py` wraps `TabNetClassifier.fit` in `warnings.catch_warnings()` with
`filterwarnings("ignore", message=...)` for exactly these two messages:

1. `"Best weights from best epoch are automatically used!"` (EarlyStopping.on_train_end;
   raised after every fit that has an eval_set, including when max_epochs is reached).
2. `"No early stopping will be performed, last training weights will be used."`
   (`_set_callbacks`; raised by the refit without eval_set, which is exactly the intended
   "train best_epoch epochs with the last weights" protocol).
