use crate::vphy::Vec3;
use crate::{KERNEL_DISPATCH_SIZE, KernelResult, VGpuContext};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Vec3,
}

impl Sphere {
    pub fn intersect(&self, ray: &Ray) -> Option<f32> {
        let oc = ray.origin.sub(self.center);
        let a = ray.dir.dot(ray.dir);
        let b = 2.0 * oc.dot(ray.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            None
        } else {
            Some((-b - discriminant.sqrt()) / (2.0 * a))
        }
    }
}

/// Octahedral Mapping: 3D Unit Vector -> 2D [0,1]
pub fn octahedral_encode(v: Vec3) -> (f32, f32) {
    let l1 = v.x.abs() + v.y.abs() + v.z.abs();
    let mut res = Vec3::new(v.x / l1, v.y / l1, v.z / l1);
    if v.z < 0.0 {
        let x = res.x;
        let y = res.y;
        res.x = (1.0 - y.abs()) * x.signum();
        res.y = (1.0 - x.abs()) * y.signum();
    }
    (res.x * 0.5 + 0.5, res.y * 0.5 + 0.5)
}

/// vRender Signature: Voxel + Octahedral Direction + Material
pub fn generate_render_signature(pos: Vec3, dir: Vec3, material_id: u32) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};

    // Voxel (10cm grid)
    let vx = (pos.x * 10.0) as i32;
    let vy = (pos.y * 10.0) as i32;
    let vz = (pos.z * 10.0) as i32;
    vx.hash(&mut s);
    vy.hash(&mut s);
    vz.hash(&mut s);

    // Direction (Octahedral 8-bit quantization per axis)
    let (u, v) = octahedral_encode(dir);
    let qu = (u * 255.0) as u8;
    let qv = (v * 255.0) as u8;
    qu.hash(&mut s);
    qv.hash(&mut s);

    material_id.hash(&mut s);
    s.finish()
}

pub fn vrender_trace(ctx: &mut VGpuContext, ray: &Ray, spheres: &[Sphere], depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::zero();
    }

    // 1. Manifold Recall (The Inductive Shunt)
    // We incorporate depth into the signature to shunt multi-bounce paths
    let sig = generate_render_signature(ray.origin, ray.dir, 0);
    let path_sig = sig ^ (depth as u64).wrapping_mul(0x517cc1b727220a95);

    if let Some(res) = ctx.inductor.recall(path_sig, path_sig) {
        if res.data.len() >= 3 {
            return Vec3::new(res.data[0], res.data[1], res.data[2]);
        }
    }

    // 2. Fallback: Ray Casting
    let mut closest_t = f32::MAX;
    let mut hit_sphere = None;

    for s in spheres {
        if let Some(t) = s.intersect(ray) {
            if t > 0.001 && t < closest_t {
                closest_t = t;
                hit_sphere = Some(s);
            }
        }
    }

    let color = if let Some(sphere) = hit_sphere {
        let hit_pt = ray.origin.add(ray.dir.mul(closest_t));
        let normal = hit_pt.sub(sphere.center).normalize();

        // Recursion: Secondary Ray for Reflection
        let reflected_dir = ray
            .dir
            .sub(normal.mul(2.0 * ray.dir.dot(normal)))
            .normalize();
        let reflected_ray = Ray {
            origin: hit_pt.add(normal.mul(0.001)),
            dir: reflected_dir,
        };
        let reflected_color = vrender_trace(ctx, &reflected_ray, spheres, depth - 1);

        // Simple PBR-lite integration
        let light_dir = Vec3::new(1.0, 1.0, 1.0).normalize();
        let diff = normal.dot(light_dir).max(0.1);
        sphere
            .color
            .mul(diff)
            .mul(0.5)
            .add(reflected_color.mul(0.5))
    } else {
        // Sky Color
        let t = 0.5 * (ray.dir.y + 1.0);
        Vec3::new(1.0, 1.0, 1.0)
            .mul(1.0 - t)
            .add(Vec3::new(0.5, 0.7, 1.0).mul(t))
    };

    // 3. Induction (Converge and Learn)
    let mut data = vec![0.0; KERNEL_DISPATCH_SIZE];
    data[0] = color.x;
    data[1] = color.y;
    data[2] = color.z;
    ctx.inductor.induct(
        path_sig,
        path_sig,
        KernelResult {
            data: Arc::from(data),
        },
    );

    color
}
