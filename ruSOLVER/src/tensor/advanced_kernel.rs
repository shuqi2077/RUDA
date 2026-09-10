// SPDX-License-Identifier: Apache-2.0
//! One-thread-per-system FP32 baselines. No host numerical fallback, barriers,
//! dynamic local allocation, or unsupported GPU f64 assumptions.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
#[cube(launch)]
pub(super)fn lu_solve(a:&Array<f32>,b:&Array<f32>,lu:&mut Array<f32>,x:&mut Array<f32>,piv:&mut Array<i32>,info:&mut Array<i32>,
order:u32,rhs_count:u32,atol:f32,rtol:f32,#[comptime]_source:String){
    let sys=ABSOLUTE_POS;if sys>=info.len(){terminate!();}
    let n=order as usize;let nr=rhs_count as usize;let base=sys*n*n;let rb=sys*n*nr;let mut code=0i32;let mut scale=0.0f32;
    for i in 0..n*n{let v=a[base+i];if v.is_nan()||v.is_inf(){code=-1;}scale=f32::max(scale,v.abs());lu[base+i]=v;}
    for i in 0..n*nr{let v=b[rb+i];x[rb+i]=v;if v.is_nan()||v.is_inf(){code=-1;}}
    for i in 0..n{piv[sys*n+i]=i as i32;}
    if code==0{if scale==0.0{code=1;}else{
        for i in 0..n*n{lu[base+i]=lu[base+i]/scale;}
        for i in 0..n*nr{x[rb+i]=x[rb+i]/scale;if x[rb+i].is_inf()||x[rb+i].is_nan(){code=-3;}}
    }}
    if code==0{for k in 0..n{if code==0{
        let mut p=k;let mut best=lu[base+k*n+k].abs();
        for i in k+1..n{let v=lu[base+i*n+k].abs();if v>best{best=v;p=i;}}
        if best<=f32::max(atol/scale,rtol){code=(k+1)as i32;}else{
            piv[sys*n+k]=p as i32;
            if p!=k{for j in 0..n{let v=lu[base+k*n+j];lu[base+k*n+j]=lu[base+p*n+j];lu[base+p*n+j]=v;}
                for c in 0..nr{let v=x[rb+k*nr+c];x[rb+k*nr+c]=x[rb+p*nr+c];x[rb+p*nr+c]=v;}}
            for i in k+1..n{let r=lu[base+i*n+k]/lu[base+k*n+k];lu[base+i*n+k]=r;
                if r.is_inf()||r.is_nan(){code=-3;}
                for j in k+1..n{let v=lu[base+i*n+j]-r*lu[base+k*n+j];lu[base+i*n+j]=v;if v.is_inf()||v.is_nan(){code=-3;}}}
        }
    }}}
    if code==0{for c in 0..nr{
        for i in 0..n{let mut v=x[rb+i*nr+c];for j in 0..i{v=v-lu[base+i*n+j]*x[rb+j*nr+c];}x[rb+i*nr+c]=v;}
        let mut end=n;while end>0{let i=end-1;let mut v=x[rb+i*nr+c];for j in i+1..n{v=v-lu[base+i*n+j]*x[rb+j*nr+c];}
            v=v/lu[base+i*n+i];x[rb+i*nr+c]=v;if v.is_nan()||v.is_inf(){code=-3;}end-=1;}
    }}
    if code==0{for i in 0..n{for j in i..n{let v=lu[base+i*n+j]*scale;lu[base+i*n+j]=v;if v.is_inf()||v.is_nan(){code=-3;}}}}
    if code!=0{for i in 0..n*n{lu[base+i]=0.0;}for i in 0..n*nr{x[rb+i]=0.0;}for i in 0..n{piv[sys*n+i]=-1;}}
    info[sys]=code;
}
#[cube(launch)]
pub(super)fn qr(a:&Array<f32>,q:&mut Array<f32>,r:&mut Array<f32>,work:&mut Array<f32>,tau:&mut Array<f32>,info:&mut Array<i32>,
rows:u32,cols:u32,atol:f32,rtol:f32,#[comptime]_source:String){
    let sys=ABSOLUTE_POS;if sys>=info.len(){terminate!();}let m=rows as usize;let n=cols as usize;
    let base=sys*m*n;let rb=sys*n*n;let mut code=0i32;let mut scale=0.0f32;
    for i in 0..m*n{let v=a[base+i];work[base+i]=v;q[base+i]=0.0;scale=f32::max(scale,v.abs());if v.is_nan()||v.is_inf(){code=-1;}}
    for i in 0..n*n{r[rb+i]=0.0;}for i in 0..n{tau[sys*n+i]=0.0;q[base+i*n+i]=1.0;}
    if code==0{if scale==0.0{code=1;}else{for i in 0..m*n{work[base+i]=work[base+i]/scale;}}}
    if code==0{for k in 0..n{if code==0{
        let mut local=0.0f32;for i in k..m{local=f32::max(local,work[base+i*n+k].abs());}
        let mut sum=0.0f32;if local>0.0{for i in k..m{let z=work[base+i*n+k]/local;sum+=z*z;}}
        let norm=local*sum.sqrt();
        if norm<=f32::max(atol/scale,rtol){code=(k+1)as i32;}else{
            let first=work[base+k*n+k];let mut sign=1.0f32;if first<0.0{sign=-1.0;}
            let divisor=first/norm+sign;let t=1.0+first.abs()/norm;tau[sys*n+k]=t;
            for i in k+1..m{work[base+i*n+k]=(work[base+i*n+k]/norm)/divisor;}
            work[base+k*n+k]=-sign*norm;
            for j in k+1..n{let mut dot=work[base+k*n+j];for i in k+1..m{dot+=work[base+i*n+k]*work[base+i*n+j];}dot*=t;
                work[base+k*n+j]-=dot;for i in k+1..m{work[base+i*n+j]-=work[base+i*n+k]*dot;}}
        }
    }}}
    if code==0{let mut end=n;while end>0{let k=end-1;let t=tau[sys*n+k];for j in 0..n{
        let mut dot=q[base+k*n+j];for i in k+1..m{dot+=work[base+i*n+k]*q[base+i*n+j];}dot*=t;
        q[base+k*n+j]-=dot;for i in k+1..m{q[base+i*n+j]-=work[base+i*n+k]*dot;}
    }end-=1;}
        for i in 0..n{for j in i..n{r[rb+i*n+j]=work[base+i*n+j]*scale;}}
        for i in 0..m*n{if q[base+i].is_nan()||q[base+i].is_inf(){code=-3;}}
        for i in 0..n*n{if r[rb+i].is_nan()||r[rb+i].is_inf(){code=-3;}}
    }
    if code!=0{for i in 0..m*n{q[base+i]=0.0;}for i in 0..n*n{r[rb+i]=0.0;}}
    info[sys]=code;
}
#[cube(launch)]
pub(super)fn eigen(a:&Array<f32>,values:&mut Array<f32>,v:&mut Array<f32>,w:&mut Array<f32>,info:&mut Array<i32>,sweeps:&mut Array<i32>,
order:u32,max_sweeps:u32,atol:f32,rtol:f32,symmetry_tol:f32,#[comptime]_source:String){
    let sys=ABSOLUTE_POS;if sys>=info.len(){terminate!();}let n=order as usize;let base=sys*n*n;let mut code=0i32;let mut scale=0.0f32;
    for i in 0..n*n{let z=a[base+i];scale=f32::max(scale,z.abs());w[base+i]=z;v[base+i]=0.0;if z.is_nan()||z.is_inf(){code=-1;}}
    for i in 0..n{v[base+i*n+i]=1.0;values[sys*n+i]=0.0;}
    let mut norm2=0.0f32;
    if code==0&&scale>0.0{for i in 0..n*n{w[base+i]=w[base+i]/scale;norm2+=w[base+i]*w[base+i];}
        for i in 0..n{for j in 0..i{if (w[base+i*n+j]-w[base+j*n+i]).abs()>symmetry_tol{code=-2;}w[base+j*n+i]=w[base+i*n+j];}}}
    let mut iter=0u32;let mut done=false;
    if code==0{let mut target=0.0f32;if scale>0.0{target=f32::max(atol/scale,rtol*norm2.sqrt());}
        while !done&&code==0{
            let mut off=0.0f32;for i in 0..n{for j in i+1..n{let z=w[base+i*n+j];off+=2.0*z*z;}}
            if off.sqrt()<=target||scale==0.0{done=true;}
            else if iter>=max_sweeps{code=-4;}
            else{for p in 0..n{for q in p+1..n{
                let apq=w[base+p*n+q];if apq!=0.0{
                    let delta=0.5*(w[base+q*n+q]-w[base+p*n+p]);let mut t=1.0f32;
                    if delta!=0.0{let mag=(delta*delta+apq*apq).sqrt();let mut denom=delta+mag;if delta<0.0{denom=delta-mag;}t=apq/denom;}
                    let c=1.0/(1.0+t*t).sqrt();let s=t*c;
                    w[base+p*n+p]-=t*apq;w[base+q*n+q]+=t*apq;w[base+p*n+q]=0.0;w[base+q*n+p]=0.0;
                    for k in 0..n{if k!=p&&k!=q{let x=w[base+k*n+p];let y=w[base+k*n+q];let xp=c*x-s*y;let yq=s*x+c*y;
                        w[base+k*n+p]=xp;w[base+p*n+k]=xp;w[base+k*n+q]=yq;w[base+q*n+k]=yq;}
                        let x=v[base+k*n+p];let y=v[base+k*n+q];v[base+k*n+p]=c*x-s*y;v[base+k*n+q]=s*x+c*y;
                    }
                }
            }}iter+=1;}
        }
    }
    if code==0{for i in 0..n{values[sys*n+i]=w[base+i*n+i]*scale;}
        for i in 0..n{let mut p=i;for j in i+1..n{if values[sys*n+j]<values[sys*n+p]{p=j;}}
            if p!=i{let z=values[sys*n+i];values[sys*n+i]=values[sys*n+p];values[sys*n+p]=z;
                for k in 0..n{let z=v[base+k*n+i];v[base+k*n+i]=v[base+k*n+p];v[base+k*n+p]=z;}}}
        for i in 0..n{if values[sys*n+i].is_inf()||values[sys*n+i].is_nan(){code=-3;}}
        for i in 0..n*n{if v[base+i].is_inf()||v[base+i].is_nan(){code=-3;}}
    }
    if code!=0{for i in 0..n{values[sys*n+i]=0.0;}for i in 0..n*n{v[base+i]=0.0;}}
    info[sys]=code;sweeps[sys]=iter as i32;
}
#[cube(launch)]
pub(super)fn cg(a:&Array<f32>,b:&Array<f32>,x:&mut Array<f32>,scratch:&mut Array<f32>,info:&mut Array<i32>,iterations:&mut Array<i32>,residual:&mut Array<f32>,
order:u32,max_iterations:u32,atol:f32,rtol:f32,symmetry_tol:f32,jacobi:u32,#[comptime]_source:String){
    let sys=ABSOLUTE_POS;if sys>=info.len(){terminate!();}let n=order as usize;let ab=sys*n*n;let vb=sys*n;let sb=sys*4*n;
    let mut code=0i32;let mut scale_a=0.0f32;let mut scale_b=0.0f32;
    for i in 0..n*n{let z=a[ab+i];scale_a=f32::max(scale_a,z.abs());if z.is_nan()||z.is_inf(){code=-1;}}
    for i in 0..n{let z=b[vb+i];scale_b=f32::max(scale_b,z.abs());if z.is_nan()||z.is_inf(){code=-1;}x[vb+i]=0.0;}
    for i in 0..4*n{scratch[sb+i]=0.0;}
    if code==0{if scale_a==0.0{code=-5;}else{for i in 0..n{
        if a[ab+i*n+i]<=0.0{code=-5;}
        for j in 0..i{if (a[ab+i*n+j]/scale_a-a[ab+j*n+i]/scale_a).abs()>symmetry_tol{code=-2;}}
    }}}
    let mut rho=0.0f32;let mut norm_b=0.0f32;let mut rn=0.0f32;let mut target=0.0f32;let mut iter=0u32;let mut done=scale_b==0.0;
    if code==0&&!done{for i in 0..n{let r=b[vb+i]/scale_b;let mut z=r;if jacobi!=0{z=r/(a[ab+i*n+i]/scale_a);}
        scratch[sb+i]=r;scratch[sb+n+i]=z;scratch[sb+2*n+i]=z;rho+=r*z;norm_b+=r*r;
    }norm_b=norm_b.sqrt();rn=norm_b;target=f32::max(atol/scale_b,rtol*norm_b);if rn<=target{done=true;}}
    while code==0&&!done&&iter<max_iterations{
        let mut curvature=0.0f32;for i in 0..n{let mut v=0.0f32;for j in 0..n{v+=(a[ab+i*n+j]/scale_a)*scratch[sb+2*n+j];}
            scratch[sb+3*n+i]=v;curvature+=scratch[sb+2*n+i]*v;}
        if curvature.is_nan()||curvature.is_inf()||rho.is_nan()||rho.is_inf(){code=-3;}
        else if curvature<=0.0||rho<=0.0{code=-5;}
        else{
            let alpha=rho/curvature;let mut rr=0.0f32;
            for i in 0..n{x[vb+i]+=alpha*scratch[sb+2*n+i];scratch[sb+i]-=alpha*scratch[sb+3*n+i];rr+=scratch[sb+i]*scratch[sb+i];}
            rn=rr.sqrt();iter+=1;let replace=iter%32==0||rn<=target||iter==max_iterations;
            if replace{rr=0.0;for i in 0..n{let mut v=0.0f32;for j in 0..n{v+=(a[ab+i*n+j]/scale_a)*x[vb+j];}
                let r=b[vb+i]/scale_b-v;scratch[sb+i]=r;rr+=r*r;}rn=rr.sqrt();}
            if rn.is_nan()||rn.is_inf(){code=-3;}
            else if rn<=target{done=true;}
            else{let mut next=0.0f32;for i in 0..n{let r=scratch[sb+i];let mut z=r;if jacobi!=0{z=r/(a[ab+i*n+i]/scale_a);}
                scratch[sb+n+i]=z;next+=r*z;}
                if next<=0.0{code=-5;}else{let beta=next/rho;for i in 0..n{
                    if replace{scratch[sb+2*n+i]=scratch[sb+n+i];}else{scratch[sb+2*n+i]=scratch[sb+n+i]+beta*scratch[sb+2*n+i];}}
                    rho=next;}
            }
        }
    }
    if code==0&&!done{code=-4;}
    if code==0||code==-4{for i in 0..n{let v=(x[vb+i]*scale_b)/scale_a;x[vb+i]=v;if v.is_nan()||v.is_inf(){code=-3;}}}
    let reported=rn*scale_b;if reported.is_nan()||reported.is_inf(){code=-3;}
    if code!=0&&code!=-4{for i in 0..n{x[vb+i]=0.0;}}
    info[sys]=code;iterations[sys]=iter as i32;residual[sys]=reported;
}
