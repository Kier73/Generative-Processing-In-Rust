/// Hilbert Curve: Space-Filling Curve for Manifold Locality
/// Converts 2D (x, y) coordinates to 1D index (d) to preserve structural locality.
pub struct HilbertMap;

impl HilbertMap {
    /// Convert (x, y) coordinates to 1D index.
    /// 'n' must be a power of 2.
    /// Upgraded to u128 to support exascale (10^24+) depths.
    pub fn xy_to_d(mut x: u64, mut y: u64, n: u64) -> u128 {
        let mut d: u128 = 0;
        let mut s = n / 2;
        while s > 0 {
            let rx = if (x & s) > 0 { 1 } else { 0 };
            let ry = if (y & s) > 0 { 1 } else { 0 };
            d += (s as u128) * (s as u128) * ((3 * rx) ^ ry) as u128;

            // Mask out the bits for this quadrant
            x &= s - 1;
            y &= s - 1;

            // Rotate/Flip within the quadrant
            Self::rot(s, &mut x, &mut y, rx, ry);

            s /= 2;
        }
        d
    }

    /// Rotate/flip the quadrant appropriately
    fn rot(n: u64, x: &mut u64, y: &mut u64, rx: u64, ry: u64) {
        if ry == 0 {
            if rx == 1 {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            // Swap x and y
            let t = *x;
            *x = *y;
            *y = t;
        }
    }
}
