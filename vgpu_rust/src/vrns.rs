// vMath Synthesis: vgpu_rust/src/vrns.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsScalar {
    pub residues: Vec<u16>,
    pub moduli_count: usize,
}

const MODULI: [u64; 4] = [211, 223, 227, 229];

impl RnsScalar {
    pub fn new(val: u64, _bit_depth: u32) -> Self {
        let count = 4;
        let mut res = Vec::with_capacity(count);
        for i in 0..count {
            res.push((val % MODULI[i]) as u16);
        }
        RnsScalar {
            residues: res,
            moduli_count: count,
        }
    }

    pub fn reconstruct(&self) -> u64 {
        let mut m_prod: u128 = 1;
        for &m in &MODULI {
            m_prod *= m as u128;
        }

        let mut sum: u128 = 0;
        for (i, &m) in MODULI.iter().enumerate() {
            let m_i = m_prod / (m as u128);
            let y_i = mod_inverse(m_i as i128, m as i128) as u128;
            let r_i = self.residues[i] as u128;

            // Formula: sum = sum + r_i * m_i * y_i
            let term = (r_i * (m_i % m_prod)) % m_prod;
            let term = (term * (y_i % m_prod)) % m_prod;
            sum = (sum + term) % m_prod;
        }
        sum as u64
    }
}

fn mod_inverse(a: i128, m: i128) -> i128 {
    let mut t = 0;
    let mut newt = 1;
    let mut r = m;
    let mut newr = a % m;

    while newr != 0 {
        let q = r / newr;
        (t, newt) = (newt, t - q * newt);
        (r, newr) = (newr, r - q * newr);
    }
    if r > 1 {
        panic!("m_i is not invertible");
    }
    if t < 0 { t + m } else { t }
}
