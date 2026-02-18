# Arbitrary Uniformity

## Mathematic Formalization
To ensure Bit-Exact Dissonance Checks across different CPU vendors, we must abandon host-level `sin()` and `cos()` in favor of deterministic **CORDIC** (Coordinate Rotation Digital Computer) or **Chebyshev Polynomial** approximations.

**Chebyshev Approximation for Sin(x):**
$Sin(x) \approx \sum_{n=0}^{N} c_n T_n(x)$
Where $c_n$ are pre-computed constants and $T_n(x)$ are Chebyshev polynomials. This is executed using only deterministic `FAdd` and `FMul`, ensuring identical bit-patterns on Intel, AMD, and ARM.

## Code Scaffold

```rust
pub struct DeterministicMath;

impl DeterministicMath {
    /// Bit-exact sine approximation using Chebyshev polynomials
    pub fn stable_sin(x: f32) -> f32 {
        let x_clamped = x % (2.0 * std::f32::consts::PI);
        // Pre-computed Chebyshev coefficients
        const C: [f32; 5] = [/* ... */];
        // Use Horner's method for stable evaluation
        let mut res = C[4];
        for i in (0..4).rev() {
            res = res * x_clamped + C[i];
        }
        res
    }
}
```
