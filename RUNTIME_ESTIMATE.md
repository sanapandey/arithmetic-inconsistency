# Runtime Estimates for 100 Samples

## Configuration
- Model: Llama-2-7b-hf (7B parameters)
- Samples: 100 per dataset
- Datasets: 4 (numeric, english, spanish, italian)
- Total: 400 counterfactual pairs

## Runtime Breakdown

### Phase 1: Evaluation (100 samples × 4 datasets)

**Per sample:**
- Process x (original): 1 forward pass
- Process x' (corrupted): 1 forward pass
- **Total: 2 forward passes per sample**

**Total forward passes:**
- 100 samples × 2 passes × 4 datasets = **800 forward passes**

**Time per forward pass:**
- **CPU**: ~2-5 seconds (depends on CPU speed)
- **MPS (Apple Silicon GPU)**: ~0.5-1 second (if it works)
- **CUDA (NVIDIA GPU)**: ~0.3-0.5 seconds

**Evaluation time:**
- **CPU**: 800 × 3s = ~2400s = **~40 minutes**
- **MPS**: 800 × 0.7s = ~560s = **~9 minutes** (if compatible)
- **CUDA**: 800 × 0.4s = ~320s = **~5 minutes**

### Phase 2: Activation Patching (100 samples × ~10-20 layers)

**Per sample:**
- Patch at each layer: 1 forward pass per layer
- **Total: ~10-20 forward passes per sample** (depending on how many layers we test)

**Total forward passes:**
- 100 samples × 15 layers (average) × 4 datasets = **6,000 forward passes**

**Patching time:**
- **CPU**: 6000 × 3s = ~18000s = **~5 hours**
- **MPS**: 6000 × 0.7s = ~4200s = **~70 minutes** (if compatible)
- **CUDA**: 6000 × 0.4s = ~2400s = **~40 minutes**

### Phase 3: Analysis & Visualization

- **~2-5 minutes** (regardless of device)

## Total Runtime Estimates

### On CPU (macOS, after my fixes)
- Evaluation: ~40 minutes
- Patching: ~5 hours
- Analysis: ~5 minutes
- **Total: ~5.75 hours**

### On MPS (if you undo fixes and it works)
- Evaluation: ~9 minutes
- Patching: ~70 minutes
- Analysis: ~5 minutes
- **Total: ~1.4 hours** (much faster!)

### On CUDA (Linux with NVIDIA GPU)
- Evaluation: ~5 minutes
- Patching: ~40 minutes
- Analysis: ~5 minutes
- **Total: ~50 minutes**

## Important Notes

### If You Undo the MPS Fixes:

**Risk:**
- If macOS < 14.0: Will crash with `isin_Tensor_Tensor_out` error
- If macOS ≥ 14.0: Should work and be much faster

**To check your macOS version:**
```bash
sw_vers
```

**If macOS 14.0+:**
- You can safely undo the fixes
- Runtime will be ~1.4 hours instead of ~5.75 hours
- **4x faster!**

**If macOS < 14.0:**
- Keep the fixes (CPU mode)
- Or upgrade macOS to 14.0+
- Or use a Linux machine with GPU

## Recommendation

**For 100 samples:**

1. **Check macOS version first:**
   ```bash
   sw_vers
   ```

2. **If macOS 14.0+:**
   - Undo the MPS fixes (remove the MPS disable code)
   - Expected runtime: **~1.5 hours**
   - Much faster!

3. **If macOS < 14.0:**
   - Keep the fixes
   - Expected runtime: **~6 hours** on CPU
   - Or consider using fewer layers for patching to reduce time

4. **Alternative: Reduce layers tested**
   - Instead of testing all layers, test top 5-10 most important ones
   - Reduces patching time from ~5 hours to ~1-2 hours
   - Still answers all 6 questions

## Quick Test

To see if MPS works on your system:
```bash
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('MPS available')
    x = torch.tensor([1, 2, 3], dtype=torch.long)
    y = torch.tensor([2, 3], dtype=torch.long)
    try:
        result = torch.isin(x, y)
        print('MPS isin works! You can use MPS.')
    except:
        print('MPS isin fails. Use CPU.')
else:
    print('MPS not available')
"
```

