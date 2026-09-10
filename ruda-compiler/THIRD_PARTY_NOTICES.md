# Third-party notice

The API naming and operation semantics in this crate are derived from
DeepSeek's DeepGEMM 2.6.1, commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`:

https://github.com/deepseek-ai/DeepGEMM

DeepGEMM is distributed under the MIT License:

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Arm optimized-routines: single-precision exponential and logarithm

`src/ptx/exp.rs` adapts the 32-entry table, range reduction and polynomial
from Arm optimized-routines `math/expf.c` and `math/exp2f_data.c` into Rust
PTX emission. Source files accessed on 2026-09-08:

https://github.com/ARM-software/optimized-routines/blob/master/math/expf.c

https://github.com/ARM-software/optimized-routines/blob/master/math/exp2f_data.c

`src/ptx/log.rs` adapts the 16-entry table, subnormal normalization, range
reduction and polynomial from the following files, accessed on 2026-09-08:

https://github.com/ARM-software/optimized-routines/blob/master/math/logf.c

https://github.com/ARM-software/optimized-routines/blob/master/math/logf_data.c

https://github.com/ARM-software/optimized-routines/blob/master/LICENSE

The MIT option of the upstream dual license is used for these adaptations.

`src/ptx/trigonometry.rs` adapts range reduction and polynomials from
`math/sinf.c`, `math/cosf.c`, `math/sincosf.h` and `math/sincosf_data.c`,
accessed on 2026-09-09, under the same MIT option:

https://github.com/ARM-software/optimized-routines/tree/master/math

Copyright (c) 2018-2024, Arm Limited.

Copyright (c) 2017-2025, Arm Limited. (`expf.c`)

Copyright (c) 2017-2018, Arm Limited. (`exp2f_data.c`)

Copyright (c) 2017-2025, Arm Limited. (`logf.c`)

Copyright (c) 2017-2024, Arm Limited. (`logf_data.c`)

MIT License

Copyright (c) 1999-2022, Arm Limited.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## OpenLibm: single-precision hyperbolic functions, expm1 and log1p

`src/ptx/tanh.rs` and `src/ptx/expm1.rs` adapt the algorithms in OpenLibm
`src/s_tanhf.c` and `src/s_expm1f.c` into Rust PTX emission. Source files
accessed on 2026-09-08:

https://github.com/JuliaMath/openlibm/blob/master/src/s_tanhf.c

https://github.com/JuliaMath/openlibm/blob/master/src/s_expm1f.c

`src/ptx/log1p.rs` adapts the range reduction, correction term and polynomial
from the following file, accessed on 2026-09-08:

https://github.com/JuliaMath/openlibm/blob/master/src/s_log1pf.c

`src/ptx/atanh.rs` adapts the inverse hyperbolic tangent algorithm from the
following file, accessed on 2026-09-08:

https://github.com/JuliaMath/openlibm/blob/master/src/e_atanhf.c

`src/ptx/inverse_hyperbolic.rs` adapts the piecewise asinh/acosh formulas
from the following files, accessed on 2026-09-08, using the existing Ruda
logarithm, log1p and square-root emitters:

https://github.com/JuliaMath/openlibm/blob/master/src/s_asinhf.c

https://github.com/JuliaMath/openlibm/blob/master/src/e_acoshf.c

`src/ptx/hyperbolic.rs` adapts the piecewise sinh/cosh formulas from the
following files, accessed on 2026-09-08. The near-overflow branch scales
the existing Arm-based exponential before F32 conversion; it does not
use OpenLibm's `k_expf.c` implementation.

https://github.com/JuliaMath/openlibm/blob/master/src/e_sinhf.c

https://github.com/JuliaMath/openlibm/blob/master/src/e_coshf.c

Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
