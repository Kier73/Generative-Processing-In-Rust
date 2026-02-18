use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, Debug)]
pub struct KernelResult {
    pub data: Arc<[f32]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ManifoldKey {
    pub sig: u64,
    pub hash: u64,
}

pub struct VInductor {
    pub manifold: RwLock<HashMap<ManifoldKey, KernelResult>>,
    pub observations: RwLock<HashMap<ManifoldKey, Vec<f32>>>,
    pub id: u32,
    pub depth: u32,
}

fn main() {
    println!(
        "Size of VInductor (Main Struct): {} bytes",
        std::mem::size_of::<VInductor>()
    );
    println!(
        "Size of ManifoldKey: {} bytes",
        std::mem::size_of::<ManifoldKey>()
    );
    println!(
        "Size of RwLock<HashMap<...>>: {} bytes",
        std::mem::size_of::<RwLock<HashMap<ManifoldKey, KernelResult>>>()
    );

    // Theoretical comparison
    println!("\n--- Memory Usage Comparison (per Inductor) ---");
    println!("Old (Fixed 64k Array): ~6 MB");
    println!(
        "New (Sparse HashMap):  ~{} bytes (Empty)",
        std::mem::size_of::<VInductor>()
    );
    println!(
        "New (1000 laws):       ~{} KB (Estimated)",
        (1000 * (std::mem::size_of::<ManifoldKey>() + std::mem::size_of::<KernelResult>() + 16))
            / 1024
    );
}
