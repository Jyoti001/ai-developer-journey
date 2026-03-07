# Week 1 — Day 2: Gradient Descent from Scratch (NumPy)

## What I built
Implemented gradient descent from scratch using NumPy on a 
synthetic dataset y = 2x + 3 + noise. Model correctly converged 
to w≈1.89, b≈3.6.

## Key learnings

### 1. Why gradient for w includes x but not b
When we differentiate loss w.r.t w, the chain rule brings down 
x from wx. When we differentiate w.r.t b, the wx term vanishes 
(no b in it) and b differentiates to 1. So:
  dLoss/dw = -2x(y - ypred)
  dLoss/db = -2(y - ypred)

### 2. Gradient Explosion
When learning rate is too high (tested lr=0.1), each update 
overshoots the minimum, landing on a steeper slope, causing 
the next update to be even larger. Weights diverged to 10^152.
Fix: use smaller learning rate or normalise inputs.

### 3. Input Normalisation
When x ranged 0→1000 instead of 0→1, gradients exploded even 
with small lr=0.0001. Large input values amplify gradients.
Always normalise inputs to 0→1 or mean=0, std=1.

### 4. np.random.seed(42)
Fixes the random number sequence so experiments are reproducible.
Without it, different noise each run makes it impossible to know
if results changed due to code or randomness.

### 5. Weights, bias, loss are scalars
Gradient must be summed across all data points using np.sum()
before updating. Without np.sum(), w becomes an array — silent 
bug that breaks the entire training loop.

## Experiments run
| Experiment | Result | Learning |
|---|---|---|
| epochs=110 | w=1.81, b=3.32 | Didn't fully converge |
| epochs=1000 | w=2.007, b=3.48 | Better convergence |
| lr=0.1 | w=10^152 | Gradient explosion |
| x range 0→1000 | Diverged at epoch 12 | Normalisation matters |

## Bugs I made and fixed
1. x range 0→1000 → caused explosion → fixed to 0→1
2. Missing np.sum() → w became array → added np.sum()
3. Variable name collision (b for data AND bias) → renamed clearly
4. No plot labels → added xlabel, ylabel, title

### 6. Variable Naming Collision — Silent Bug
In version 1, I used 'b' for both the data constant (b = np.ones(100)*3) 
and the bias parameter (b = 0). Python silently overwrote the first b 
with the second — no error, no warning, just wrong behaviour.

Rule: always use distinct names for data variables vs model parameters.
  ❌ b = 3  then later  b = 0   → collision
  ✅ b_constant = 3  and  bias = 0  → clear and safe

This matters in PyTorch too — confusing tensor names with 
scalar parameters causes the same class of silent bugs.