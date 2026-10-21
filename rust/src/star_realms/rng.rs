//! CPython's `random.Random`, as far as the engine uses it: seeding from an
//! integer, `getrandbits` and `sample(cards, len(cards))`.
//!
//! The Python engine shuffles with `random.Random(seed).sample(cards,
//! len(cards))` and draws the next seed with `getrandbits(64)`
//! (`rules._shuffle`). Reproducing CPython's MT19937, its seeding by array and
//! its `_randbelow` exactly makes a seed deal the same game in both engines,
//! so the differential tests compare every state exactly (docs/star-realms.md,
//! "Compiled engine").

const N: usize = 624;
const M: usize = 397;

/// The Mersenne Twister, seeded as CPython seeds it from a non-negative int.
pub struct Random {
    mt: [u32; N],
    index: usize,
}

impl Random {
    /// `random.Random(seed)`: CPython splits the seed's absolute value into
    /// 32-bit words, least significant first (one word, 0, for a zero seed),
    /// and seeds MT19937 with them by `init_by_array`.
    pub fn new(seed: u64) -> Self {
        let low = seed as u32;
        let high = (seed >> 32) as u32;
        let mut rng = Random {
            mt: [0; N],
            index: N,
        };
        if high == 0 {
            rng.init_by_array(&[low]);
        } else {
            rng.init_by_array(&[low, high]);
        }
        rng
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1, 0);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn generate(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for k in 0..N {
            let y = (self.mt[k] & UPPER) | (self.mt[(k + 1) % N] & LOWER);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
        }
        self.index = 0;
    }

    /// The next 32-bit output (CPython's `genrand_uint32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for `1 <= k <= 32`: the top `k` bits of one output.
    fn bits(&mut self, k: u32) -> u32 {
        debug_assert!((1..=32).contains(&k));
        self.next_u32() >> (32 - k)
    }

    /// `getrandbits(64)`: two outputs, the first the least significant.
    pub fn next_u64(&mut self) -> u64 {
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32());
        low | (high << 32)
    }

    /// `_randbelow(n)` for `n > 0`: `getrandbits(n.bit_length())`, retried
    /// until below `n`.
    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0 && n <= u32::MAX as usize);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.bits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `sample(items, len(items))`. With `k == n`, CPython always takes its
    /// pool branch: its set-size threshold, `21 + 4 ** ceil(log(3k, 4))`,
    /// exceeds `n`.
    pub fn sample<T: Copy>(&mut self, items: &[T]) -> Vec<T> {
        let n = items.len();
        let mut pool = items.to_vec();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let j = self.below(n - i);
            result.push(pool[j]);
            pool[j] = pool[n - i - 1];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Random;

    // Expected values from CPython 3.14's `random` module.

    #[test]
    fn getrandbits_matches_cpython() {
        let mut rng = Random::new(42);
        let words: Vec<u32> = (0..3).map(|_| rng.next_u32()).collect();
        assert_eq!(words, [2_746_317_213, 478_163_327, 107_420_369]);
        assert_eq!(rng.next_u64(), 5_073_395_517_033_431_291);
    }

    #[test]
    fn a_seed_over_32_bits_seeds_with_two_words() {
        assert_eq!(
            Random::new((1 << 40) + 5).next_u64(),
            9_535_518_150_286_577_956
        );
    }

    #[test]
    fn sample_matches_cpython() {
        let mut rng = Random::new(0);
        let items: Vec<u32> = (0..10).collect();
        assert_eq!(rng.sample(&items), [6, 9, 0, 2, 4, 3, 5, 1, 8, 7]);
        assert_eq!(rng.next_u64(), 16_448_235_973_081_859_711);

        let mut rng = Random::new(123_456_789_012_345_678);
        let items: Vec<u32> = (0..30).collect();
        assert_eq!(
            rng.sample(&items),
            [
                13, 22, 16, 6, 4, 21, 25, 20, 11, 28, 17, 14, 24, 15, 3, 10, 12, 9, 18, 29, 5, 19,
                23, 7, 8, 2, 0, 1, 27, 26
            ]
        );
        assert_eq!(rng.next_u64(), 15_481_149_845_397_082_302);
    }
}
