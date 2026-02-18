// vMath Synthesis: vgpu_rust/src/vgeometric.rs

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// The 11 Atomic Primitives of Computation.
/// A complete basis for expressing any logical or mathematical transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Primitive {
    ID, // Identity: f(x) = x
    CP, // Composition: f(g(x))
    RC, // Recursion: Fixed-point loops
    CH, // Choice: Branching
    RL, // Relation: Ordering (>, <, =)
    DY, // Duality: Inversion/Negation
    AB, // Abstraction: Enclosure/Scoping
    PM, // Permutation: Reordering
    AG, // Aggregation: Reduction (Sum, Max)
    PR, // Projection: Selection
    QM, // Quantum Mapping: Superposition
}

/// A specific instance of a Law (Logic Tree).
#[derive(Debug, Clone)]
pub struct LawNode {
    pub primitive: Primitive,
    pub children: Vec<LawNode>,
    pub value: Option<u64>, // For leaf nodes (literals)
}

impl LawNode {
    pub fn new(prim: Primitive) -> Self {
        LawNode {
            primitive: prim,
            children: Vec::new(),
            value: None,
        }
    }

    pub fn with_children(mut self, children: Vec<LawNode>) -> Self {
        self.children = children;
        self
    }

    pub fn signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.primitive.hash(&mut hasher);

        // Fold children signatures with Feistel-like mixing
        let mut child_hash = 0u64;
        for c in &self.children {
            let sig = c.signature();
            child_hash = feistel_mix(child_hash, sig);
        }
        child_hash.hash(&mut hasher);

        if let Some(v) = self.value {
            v.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Feistel Mix for structural entropy preservation.
/// L' = R
/// R' = L ^ F(R)
/// Here simplified to a single reversible mix step.
fn feistel_mix(l: u64, r: u64) -> u64 {
    let k = 0x9E3779B97F4A7C15u64; // Golden Ratio constant
    let f_r = (r.rotate_left(31) ^ r.rotate_right(17)).wrapping_mul(k);
    l ^ f_r
}

/// Manifold Coordinate Binding: M = L XOR I
/// Maps a Law (L) and Input (I) to a unique Latent Coordinate.
pub fn manifold_coordinate(law_sig: u64, input_hash: u64) -> u64 {
    feistel_mix(law_sig, input_hash)
}
