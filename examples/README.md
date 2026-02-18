# vGPU Examples

This directory contains scripts and demonstrations showing the vGPU in action. 

## Benchmarks
- `bench_small_scale.py`: Comparison of NumPy and vGPU for small matrix tiles.
- `bench_procedural_gen.py`: Demonstrates the speed of position-based hashing for coherent noise generation.
- `bench_vgpu_vs_numpy.py`: High-level comparison showing linear vs. geometric scaling.

## Utilities
- `verify_memory.py`: (Reserved) Memory pressure and manifold saturation test scripts.

## Usage
Most Python examples require the `vgpu_rust` shared library to be built:
```bash
cd vgpu_rust
cargo build --release
```
Then run the example:
```bash
python bench_small_scale.py
```
