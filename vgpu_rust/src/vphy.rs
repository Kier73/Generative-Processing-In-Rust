use crate::{KernelResult, VGpuContext};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
    pub fn zero() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    pub fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    pub fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 1e-6 {
            self.mul(1.0 / l)
        } else {
            Vec3::zero()
        }
    }
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub pos: Vec3,
    pub vel: Vec3,
    pub inv_mass: f32,
    pub radius: f32,
}

pub struct Contact {
    pub normal: Vec3,
    pub depth: f32,
}

/// SituationSignature: A hash that is invariant to absolute world coordinates.
/// It depends only on relative state between two interacting bodies.
pub fn generate_situation_signature(a: &RigidBody, b: &RigidBody) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};

    // Relative Position (Quantized to 1cm)
    let rel_pos = b.pos.sub(a.pos);
    let qx = (rel_pos.x * 100.0) as i32;
    let qy = (rel_pos.y * 100.0) as i32;
    let qz = (rel_pos.z * 100.0) as i32;
    qx.hash(&mut s);
    qy.hash(&mut s);
    qz.hash(&mut s);

    // Relative Velocity (Quantized to 0.1m/s)
    let rel_vel = b.vel.sub(a.vel);
    let vx = (rel_vel.x * 10.0) as i32;
    let vy = (rel_vel.y * 10.0) as i32;
    let vz = (rel_vel.z * 10.0) as i32;
    vx.hash(&mut s);
    vy.hash(&mut s);
    vz.hash(&mut s);

    s.finish()
}

/// Jacobian Fingerprint: Encodes the geometric constraint manifold between two bodies.
/// Includes the contact normal and distances, allowing the inductor to learn the PGS solution.
pub fn generate_jacobian_signature(a: &RigidBody, b: &RigidBody, normal: Vec3) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};

    // Relative Position + Normal Fingerprint
    let rel_pos = b.pos.sub(a.pos);
    ((rel_pos.x * 100.0) as i32).hash(&mut s);
    ((rel_pos.y * 100.0) as i32).hash(&mut s);
    ((rel_pos.z * 100.0) as i32).hash(&mut s);

    ((normal.x * 1000.0) as i32).hash(&mut s);
    ((normal.y * 1000.0) as i32).hash(&mut s);
    ((normal.z * 1000.0) as i32).hash(&mut s);

    s.finish()
}

/// Jacobian-Inductive Impulse Solver (PGS Foundation)
pub fn solve_collision_jacobian(
    ctx: &mut VGpuContext,
    a: &mut RigidBody,
    b: &mut RigidBody,
    _dt: f32,
) -> (Vec3, Vec3) {
    let rel_pos = b.pos.sub(a.pos);
    let dist = rel_pos.length();
    let min_dist = a.radius + b.radius;

    if dist < min_dist && dist > 1e-6 {
        let normal = rel_pos.normalize();
        let sig = generate_jacobian_signature(a, b, normal);

        // 1. Inductive Recall of the Impulse λ
        if let Some(res) = ctx.inductor.recall(sig, sig) {
            if res.data.len() >= 6 {
                return (
                    Vec3::new(res.data[0], res.data[1], res.data[2]),
                    Vec3::new(res.data[3], res.data[4], res.data[5]),
                );
            }
        }

        // 2. Fallback: Projected Gauss-Seidel 1-Step
        let _depth = min_dist - dist;
        let rel_vel = b.vel.sub(a.vel);
        let vel_along_normal = rel_vel.dot(normal);

        if vel_along_normal > 0.0 {
            return (Vec3::zero(), Vec3::zero());
        }

        let e = 0.5;
        let j = -(1.0 + e) * vel_along_normal;
        let j = j / (a.inv_mass + b.inv_mass);
        let impulse = normal.mul(j);

        let dva = impulse.mul(a.inv_mass);
        let dvb = impulse.mul(b.inv_mass);

        // 3. Induct the Jacobson Solution
        let mut data = vec![0.0; 1024];
        data[0] = dva.x;
        data[1] = dva.y;
        data[2] = dva.z;
        data[3] = dvb.x;
        data[4] = dvb.y;
        data[5] = dvb.z;

        ctx.inductor.induct(
            sig,
            sig,
            KernelResult {
                data: Arc::from(data),
            },
        );

        (dva, dvb)
    } else {
        (Vec3::zero(), Vec3::zero())
    }
}

pub fn vphy_step(ctx: &mut VGpuContext, a: &mut RigidBody, b: &mut RigidBody, dt: f32) {
    let sig = generate_situation_signature(a, b);
    let input_hash = sig; // Simplification: we use the signature as the situational seed

    // 1. Manifold Recall
    if let Some(res) = ctx.inductor.recall(sig, input_hash) {
        if res.data.len() >= 6 {
            a.vel.x += res.data[0];
            a.vel.y += res.data[1];
            a.vel.z += res.data[2];
            b.vel.x += res.data[3];
            b.vel.y += res.data[4];
            b.vel.z += res.data[5];
            return;
        }
    }

    // 2. Fallback Solver
    let (dva, dvb): (Vec3, Vec3) = solve_collision_jacobian(ctx, a, b, dt);

    // 3. Induction
    if dva.length_sq() > 0.0 || dvb.length_sq() > 0.0 {
        let mut data = vec![0.0; 1024];
        data[0] = dva.x;
        data[1] = dva.y;
        data[2] = dva.z;
        data[3] = dvb.x;
        data[4] = dvb.y;
        data[5] = dvb.z;
        ctx.inductor.induct(
            sig,
            input_hash,
            KernelResult {
                data: Arc::from(data),
            },
        );
    }

    // 4. Integrate
    a.pos = a.pos.add(a.vel.mul(dt));
    b.pos = b.pos.add(b.vel.mul(dt));
}
