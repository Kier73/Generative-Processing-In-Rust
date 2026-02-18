// vMath Synthesis: vgpu_rust/src/veigen.rs

/// Minimal Complex Number implementation for Manifold Dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    pub fn add(self, other: Self) -> Self {
        Complex::new(self.re + other.re, self.im + other.im)
    }

    pub fn sub(self, other: Self) -> Self {
        Complex::new(self.re - other.re, self.im - other.im)
    }

    pub fn mul(self, other: Self) -> Self {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn powf(self, t: f32) -> Self {
        // z^t = r^t * e^(i * theta * t)
        let r = self.norm_sq().sqrt();
        let theta = self.im.atan2(self.re);
        let new_r = r.powf(t);
        let new_theta = theta * t;
        Complex::new(new_r * new_theta.cos(), new_r * new_theta.sin())
    }
}

/// 2x2 Geometric Matrix for Manifold Ops.
#[derive(Debug, Clone, Copy)]
pub struct Mat2x2 {
    pub m: [[f32; 2]; 2],
}

impl Mat2x2 {
    pub fn det(&self) -> f32 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    pub fn trace(&self) -> f32 {
        self.m[0][0] + self.m[1][1]
    }

    /// Analytical Eigenvalue Solver.
    /// Returns (lambda1, lambda2).
    pub fn eigenvalues(&self) -> (Complex, Complex) {
        let tr = self.trace();
        let det = self.det();
        let discriminant = tr * tr - 4.0 * det;

        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            let l1 = (tr + root) * 0.5;
            let l2 = (tr - root) * 0.5;
            (Complex::new(l1, 0.0), Complex::new(l2, 0.0))
        } else {
            let root = (-discriminant).sqrt();
            let re = tr * 0.5;
            let im = root * 0.5;
            (Complex::new(re, im), Complex::new(re, -im))
        }
    }

    /// Temporal Memoization: M^t
    /// Using Cayley-Hamilton or Diagonalization.
    /// For 2x2, we can use the eigenvalues directly if diagonalizable.
    /// Simplified: If real distinct eigenvalues, M^t = P D^t P^-1.
    /// If complex, rotational dynamics.
    /// This is a placeholder for the full solver spec.
    pub fn predict_state(&self, t: f32) -> (Complex, Complex) {
        let (l1, l2) = self.eigenvalues();
        (l1.powf(t), l2.powf(t))
    }
}
