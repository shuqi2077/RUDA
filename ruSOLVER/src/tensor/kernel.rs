// SPDX-License-Identifier: Apache-2.0
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
// No shared memory or barriers: a failed system cannot strand other systems.
// Scratch/output allocations are exclusively owned by this launch. Inputs are read-only.
#[cube(launch)]
pub(super)fn cholesky_solve(a: &Array<f32>, b: &Array<f32>, lower: &mut Array<f32>,
x: &mut Array<f32>, info: &mut Array<i32>, order: u32, rhs_count: u32, shift: f32,
symmetry_atol: f32, symmetry_rtol: f32, #[comptime]_source: String){
    let system=ABSOLUTE_POS;
    if system>=info.len(){
        terminate!();
    }
    let n=order as usize;
    let nrhs=rhs_count as usize;
    let base=system*n*n;
    let rhs_base=system*n*nrhs;
    let mut code=0i32;
    for i in 0..n*n{
        lower[base+i]=0.0;
        if a[base+i].is_nan()||a[base+i].is_inf(){
            code=-1;
        }
    }
    for i in 0..n*nrhs{
        x[rhs_base+i]=0.0;
        if b[rhs_base+i].is_nan()||b[rhs_base+i].is_inf(){
            code=-1;
        }
    }
    if code==0{
        for i in 0..n{
            for j in 0..i{
                let u=a[base+i*n+j];
                let v=a[base+j*n+i];
                let s=f32::max(u.abs(), v.abs());
                if s>0.0{
                    if (u/s-v/s).abs()>f32::max(symmetry_atol/s, symmetry_rtol){
                        code=-2;
                    }
                }
            }
        }
    }
    if code==0{
        for j in 0..n{
            if code==0{
                let mut diagonal=a[base+j*n+j]+shift;
                for k in 0..j{
                    diagonal=diagonal-lower[base+j*n+k]*lower[base+j*n+k];
                }
                if diagonal.is_nan()||diagonal.is_inf(){
                    code=-3;
                }
                else if diagonal<=0.0{
                    code=(j+1) as i32;
                }
                else{
                    let pivot=diagonal.sqrt();
                    lower[base+j*n+j]=pivot;
                    for i in j+1..n{
                        let mut sum=a[base+i*n+j];
                        for k in 0..j{
                            sum=sum-lower[base+i*n+k]*lower[base+j*n+k];
                        }
                        let value=sum/pivot;
                        if value.is_nan()||value.is_inf(){
                            code=-3;
                        } else{
                            lower[base+i*n+j]=value;
                        }
                    }
                }
            }
        }
    }
    if code==0{
        for c in 0..nrhs{
            for i in 0..n{
                let mut value=b[rhs_base+i*nrhs+c];
                for k in 0..i{
                    value=value-lower[base+i*n+k]*x[rhs_base+k*nrhs+c];
                }
                value=value/lower[base+i*n+i];
                if value.is_nan()||value.is_inf(){
                    code=-3;
                } else{
                    x[rhs_base+i*nrhs+c]=value;
                }
            }
            let mut end=n;
            while end>0{
                let i=end-1;
                let mut value=x[rhs_base+i*nrhs+c];
                for k in i+1..n{
                    value=value-lower[base+k*n+i]*x[rhs_base+k*nrhs+c];
                }
                value=value/lower[base+i*n+i];
                if value.is_nan()||value.is_inf(){
                    code=-3;
                } else{
                    x[rhs_base+i*nrhs+c]=value;
                }
                end-=1;
            }
        }
    }
    if code!=0{
        for i in 0..n*n{
            lower[base+i]=0.0;
        }
        for i in 0..n*nrhs{
            x[rhs_base+i]=0.0;
        }
    }
    info[system]=code;
}
