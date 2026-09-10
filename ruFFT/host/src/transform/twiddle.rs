use alloc::vec::Vec;

const PI: f64 = core::f64::consts::PI;

pub(super) const fn const_sin(x: f64) -> f64 {
    let mut x = x;
    x = x - ((x / (2.0 * PI)) as i64 as f64) * 2.0 * PI;
    if x > PI {
        x -= 2.0 * PI;
    } else if x < -PI {
        x += 2.0 * PI;
    }
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut i = 1u32;
    while i <= 12 {
        term *= -x2 / ((2 * i) as f64 * (2 * i + 1) as f64);
        sum += term;
        i += 1;
    }
    sum
}

pub(super) const fn const_cos(x: f64) -> f64 {
    const_sin(x + PI / 2.0)
}

// ============================================================================
// Compile-time twiddle table
// ============================================================================

struct TwiddleTable<const M: usize> {
    re: [f32; M],
    im: [f32; M],
    offsets: [usize; 18],
    num_stages: usize,
}

const fn make_twiddle_table<const N: usize, const M: usize>() -> TwiddleTable<M> {
    let mut re = [0.0f32; M];
    let mut im = [0.0f32; M];
    let mut offsets = [0usize; 18];
    let num_stages = N.trailing_zeros() as usize;
    let mut pos = 0usize;
    let mut len = 2usize;
    let mut stage = 0usize;
    while stage < num_stages {
        offsets[stage] = pos;
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;
        let mut k = 0usize;
        while k < half {
            let angle = angle_step * k as f64;
            re[pos] = const_cos(angle) as f32;
            im[pos] = const_sin(angle) as f32;
            pos += 1;
            k += 1;
        }
        len <<= 1;
        stage += 1;
    }
    offsets[num_stages] = pos;
    TwiddleTable {
        re,
        im,
        offsets,
        num_stages,
    }
}

macro_rules! def_twiddle {
    ($name:ident, $n:expr) => {
        static $name: TwiddleTable<{ $n - 1 }> = make_twiddle_table::<$n, { $n - 1 }>();
    };
}

def_twiddle!(TW_2, 2);
def_twiddle!(TW_4, 4);
def_twiddle!(TW_8, 8);
def_twiddle!(TW_16, 16);
def_twiddle!(TW_32, 32);
def_twiddle!(TW_64, 64);
def_twiddle!(TW_128, 128);
def_twiddle!(TW_256, 256);
def_twiddle!(TW_512, 512);
def_twiddle!(TW_1024, 1024);
def_twiddle!(TW_2048, 2048);
def_twiddle!(TW_4096, 4096);
def_twiddle!(TW_8192, 8192);
def_twiddle!(TW_16384, 16384);
def_twiddle!(TW_32768, 32768);
def_twiddle!(TW_65536, 65536);

pub(super) enum TwiddleRef {
    Static {
        re: &'static [f32],
        im: &'static [f32],
        offsets: &'static [usize],
    },
    Owned {
        re: Vec<f32>,
        im: Vec<f32>,
        offsets: Vec<usize>,
    },
}

impl TwiddleRef {
    pub(super) fn re(&self) -> &[f32] {
        match self {
            Self::Static { re, .. } => re,
            Self::Owned { re, .. } => re,
        }
    }
    pub(super) fn im(&self) -> &[f32] {
        match self {
            Self::Static { im, .. } => im,
            Self::Owned { im, .. } => im,
        }
    }
    pub(super) fn offsets(&self) -> &[usize] {
        match self {
            Self::Static { offsets, .. } => offsets,
            Self::Owned { offsets, .. } => offsets,
        }
    }
}

pub(super) fn get_twiddles(n: usize) -> TwiddleRef {
    macro_rules! match_static {
        ($($size:expr => $table:ident),+ $(,)?) => {
            match n {
                0 | 1 => TwiddleRef::Static { re: &[], im: &[], offsets: &[0] },
                $($size => TwiddleRef::Static {
                    re: &$table.re, im: &$table.im,
                    offsets: &$table.offsets[..$table.num_stages + 1],
                },)+
                _ => {
                    let (re, im, offsets) = precompute_twiddles_runtime(n);
                    TwiddleRef::Owned { re, im, offsets }
                }
            }
        };
    }
    match_static!(
        2 => TW_2, 4 => TW_4, 8 => TW_8, 16 => TW_16,
        32 => TW_32, 64 => TW_64, 128 => TW_128, 256 => TW_256,
        512 => TW_512, 1024 => TW_1024, 2048 => TW_2048, 4096 => TW_4096,
        8192 => TW_8192, 16384 => TW_16384, 32768 => TW_32768, 65536 => TW_65536,
    )
}

fn precompute_twiddles_runtime(n: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let num_stages = n.trailing_zeros() as usize;
    let total = n - 1;
    let mut re = Vec::with_capacity(total);
    let mut im = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(num_stages + 1);
    let mut len = 2;
    for _ in 0..num_stages {
        offsets.push(re.len());
        let half = len / 2;
        let angle_step = -2.0 * core::f64::consts::PI / len as f64;
        for k in 0..half {
            let angle = angle_step * k as f64;
            re.push(const_cos(angle) as f32);
            im.push(const_sin(angle) as f32);
        }
        len <<= 1;
    }
    offsets.push(re.len());
    (re, im, offsets)
}

