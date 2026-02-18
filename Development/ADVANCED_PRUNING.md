# Gap 4: Advanced Manifold Pruning

## Mathematic Formalization
We propose a **Weighted Utility Score ($U$)** to determine law survival. Laws with the lowest score are pruned first during manifold saturation.

$U = \frac{H \cdot C}{L \cdot \Delta T}$

Where:
- $H$: **Hit Count** (Total times the law was recalled).
- $C$: **Computational Cost** (Complexity of the original shader).
- $L$: **Last Recall Latency** (Optional hardware metric).
- $\Delta T$: **Recency Decay** (Cycles since the last hit).

Laws that are expensive to compute ($C$) but called frequently ($H$) receive the highest scores and are protected from pruning.

## Code Scaffold

```rust
pub struct LawMetadata {
    hit_count: u64,
    last_hit: Instant,
    shader_complexity: u32,
}

impl LawMetadata {
    pub fn utility_score(&self) -> f64 {
        let age = self.last_hit.elapsed().as_secs_f64().max(0.001);
        (self.hit_count as f64 * self.shader_complexity as f64) / age
    }
}
```
