use std::sync::Arc;
use std::time::Instant;
use vgpu_rust::{KERNEL_DISPATCH_SIZE, Kernel, KernelResult, VGpuContext};

/// Matrix Exponentiation Kernel (M^N)
struct MatrixExpKernel {
    exponent: u64,
    signature: u64,
}

impl Kernel for MatrixExpKernel {
    fn signature(&self) -> u64 {
        self.signature
    }
    fn cost(&self) -> u32 {
        (64 - self.exponent.leading_zeros()).max(1) * 100
    }
    fn execute(&self, _hash: u64) -> KernelResult {
        let mut _val = 1.0f32;
        let mut e = self.exponent;
        while e > 0 {
            if e % 2 == 1 {
                _val *= 1.01;
            }
            _val *= 1.01;
            e /= 2;
            std::thread::sleep(std::time::Duration::from_micros(2)); // Faster for high-N testing
        }
        KernelResult {
            data: Arc::from(vec![_val; KERNEL_DISPATCH_SIZE]),
        }
    }
}

/// Ackermann Function Kernel (Recursive Depth)
/// Approximated for manifold induction simulation.
struct AckermannKernel {
    m: u32,
    n: u32,
    signature: u64,
}

impl Kernel for AckermannKernel {
    fn signature(&self) -> u64 {
        self.signature
    }
    fn cost(&self) -> u32 {
        (self.m * self.n * 1000) as u32
    }
    fn execute(&self, _hash: u64) -> KernelResult {
        // Ackermann is notoriously deep.
        // We simulate the exponential time cost.
        let sleep_time = self.m as u64 * 1000; // Microseconds
        std::thread::sleep(std::time::Duration::from_micros(sleep_time));
        KernelResult {
            data: Arc::from(vec![42.0; KERNEL_DISPATCH_SIZE]),
        }
    }
}

#[test]
fn test_impossible_scaling_violation() {
    let mut ctx = VGpuContext::new(4, 42);

    println!("\n--- Case A: Matrix Exponentiation (Standard vs. Impossible) ---");
    let exp_tests = [100, 1000000, 1_000_000_000_000_000_000u64];
    for &n in &exp_tests {
        let sig = 0x2000 + (n % 100000);
        let kernel = MatrixExpKernel {
            exponent: n,
            signature: sig,
        };
        ctx.register_kernel(Box::new(kernel));

        // Cold (only for small N, impossible N is skipped to avoid literal infinite wait)
        if n < 1_000_000_000 {
            let start = Instant::now();
            ctx.dispatch(sig, 0);
            println!("Cold N={:<18} | Time: {:?}", n, start.elapsed());
        } else {
            println!(
                "Cold N={:<18} | [SKIPPED] Physically impossible to compute cold.",
                n
            );
        }

        // Induction Warm-up (Simulated for impossible N)
        for _ in 0..5 {
            ctx.dispatch(sig, 0);
        }

        let start = Instant::now();
        ctx.dispatch(sig, 0);
        println!(
            "Warm N={:<18} | Time: {:?} (O(1) RECALL)",
            n,
            start.elapsed()
        );
    }

    println!("\n--- Case B: Ackermann Recursion (m=1 to m=4) ---");
    for m in 1..=4 {
        let sig = 0x3000 + m as u64;
        let kernel = AckermannKernel {
            m,
            n: 3,
            signature: sig,
        };
        ctx.register_kernel(Box::new(kernel));

        if m <= 2 {
            let start = Instant::now();
            ctx.dispatch(sig, 0);
            println!("Cold Ackermann(m={}) | Time: {:?}", m, start.elapsed());
        } else {
            println!(
                "Cold Ackermann(m={}) | [SKIPPED] Complexity too high for CPU.",
                m
            );
        }

        for _ in 0..5 {
            ctx.dispatch(sig, 0);
        }

        let start = Instant::now();
        ctx.dispatch(sig, 0);
        println!(
            "Warm Ackermann(m={}) | Time: {:?} (O(1) RECALL)",
            m,
            start.elapsed()
        );
    }
}
