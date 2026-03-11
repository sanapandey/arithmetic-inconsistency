# Fix: Wrong Python + NumPy Import Error

## What’s going on

- Your shell shows `(base)` (conda is active), but `python3` is running **system Python 3.8**:
  - `/Library/Frameworks/Python.framework/Versions/3.8/bin/python3`
- NumPy under that Python is broken: `PyCapsule_Import could not import module "datetime"`.
- Your project and conda env use a different Python (e.g. miniconda3 with Python 3.11). You want to use **that** one.

## Fix 1: Use conda’s Python (recommended)

Run everything with conda’s interpreter so you don’t depend on system Python:

```bash
# See which Python is used
which python
which python3

# If either points to /Library/Frameworks/... (system), use conda’s explicitly:
# Activate base and ensure conda’s bin is first
conda activate base
export PATH="$(conda info --base)/bin:$PATH"

# Verify you’re on conda’s Python
which python3
# Should be something like: /Users/sanapandey/miniconda3/bin/python3

# Then run your test
python3 -c "
import torch
x = torch.tensor([1, 2, 3], dtype=torch.long)
y = torch.tensor([2, 3], dtype=torch.long)
try:
    result = torch.isin(x, y)
    print('MPS works! You can use GPU acceleration.')
except Exception as e:
    print(f'MPS still has issues: {e}')
"
```

Or call the conda Python by path (replace with your actual conda base path if different):

```bash
~/miniconda3/bin/python3 -c "import torch; print('torch OK')"
```

Use this same Python to run your scripts:

```bash
# Example: run analysis with conda’s Python
~/miniconda3/bin/python3 eval_counterfactual.py --model meta-llama/Llama-2-7b-hf --dataset numeric --max-samples 10
# or, after fixing PATH:
python3 eval_counterfactual.py --model meta-llama/Llama-2-7b-hf --dataset numeric --max-samples 10
```

## Fix 2: Fix PATH so `python3` is conda

Add this to your `~/.zshrc` or `~/.bash_profile` (and restart the terminal or `source` it):

```bash
# Prefer conda’s Python when base is active
if [ -n "$CONDA_PREFIX" ]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
```

Then:

```bash
conda activate base
which python3   # should be under miniconda3
python3 -c "import torch; import numpy; print('OK')"
```

## Fix 3: If you must use system Python 3.8

Only if you really want to use `/Library/Frameworks/.../python3`:

```bash
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 -m pip install --force-reinstall numpy
```

Then rerun your test. Prefer Fix 1 so your project uses the same Python (conda) everywhere.

## Summary

1. Use conda’s Python: `conda activate base` and ensure `which python3` is under miniconda3 (or call `~/miniconda3/bin/python3`).
2. Run the MPS test and your analysis scripts with that Python.
3. Optionally fix PATH (Fix 2) so `python3` always points to conda when base is active.

The NumPy error should go away once you’re no longer using the broken system Python 3.8.
