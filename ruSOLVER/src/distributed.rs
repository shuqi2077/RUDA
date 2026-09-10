// SPDX-License-Identifier: Apache-2.0
//! Row-partitioned HOST FP64 preconditioned CG. Matrix rows stay partitioned;
//! search vectors are all-gathered, scalars globally reduced. No device/RDMA claim.
use crate::{SolverError,CgOptions,IterativeStatus};
use crate::numerics::{checked,dot,finite,sum_iter,zeros};
use crate::sparse_direct::validate_csr;
/// Ordered blocking communicator. All ranks call the same solve in the same order.
/// all_gather_fixed returns one equally-sized buffer PER rank, in rank order.
/// abort MUST wake pending peers (or transport must have a finite timeout).
pub trait SolverCommunicator {
    fn rank(&self)->usize;
    fn size(&self)->usize;
    fn all_gather_fixed(&self,phase:u64,local:&[f64])->Result<Vec<Vec<f64>>,SolverError>;
    fn abort(&self,reason:&str);
}
pub struct LocalCommunicator;
impl SolverCommunicator for LocalCommunicator{
    fn rank(&self)->usize{0}fn size(&self)->usize{1}
    fn all_gather_fixed(&self,_:u64,local:&[f64])->Result<Vec<Vec<f64>>,SolverError>{Ok(vec![local.to_vec()])}
    fn abort(&self,_:&str){}
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub struct RowPartition{pub global_rows:usize,pub start:usize,pub rows:usize}
impl RowPartition{
    pub fn balanced(n:usize,rank:usize,size:usize)->Result<Self,SolverError>{
        if n==0||size==0||rank>=size||n>u32::MAX as usize||size>u32::MAX as usize{return Err(SolverError::Shape("partition/rank"));}
        let q=n/size;let rem=n%size;let rows=q+usize::from(rank<rem);
        let start=rank*q+rank.min(rem);Ok(Self{global_rows:n,start,rows})
    }
}
pub struct DistributedCsr<'a>{pub partition:RowPartition,offsets:&'a[usize],columns:&'a[usize],values:&'a[f64]}
impl<'a>DistributedCsr<'a>{
    pub fn new(partition:RowPartition,offsets:&'a[usize],columns:&'a[usize],values:&'a[f64])->Result<Self,SolverError>{
        if partition.start.checked_add(partition.rows).is_none_or(|end|end>partition.global_rows){return Err(SolverError::Shape("partition range"));}
        validate_csr(partition.rows,partition.global_rows,offsets,columns,values)?;Ok(Self{partition,offsets,columns,values})
    }
    fn apply(&self,x:&[f64])->Result<Vec<f64>,SolverError>{
        if x.len()!=self.partition.global_rows{return Err(SolverError::Shape("distributed vector length"));}let mut y=zeros(self.partition.rows)?;
        for i in 0..y.len(){y[i]=sum_iter((self.offsets[i]..self.offsets[i+1]).map(|p|self.values[p]*x[self.columns[p]]))?;}Ok(y)
    }
    fn diagonal(&self)->Result<Vec<f64>,SolverError>{
        let mut d=zeros(self.partition.rows)?;for i in 0..d.len(){d[i]=sum_iter((self.offsets[i]..self.offsets[i+1]).filter(|&p|self.columns[p]==i+self.partition.start).map(|p|self.values[p]))?;
            if d[i]<=0.0{return Err(SolverError::NotPositiveDefinite{index:i+self.partition.start,pivot:d[i]});}}
        Ok(d)
    }
}
#[derive(Clone,Debug)]pub struct DistributedCgReport{
    pub local_solution:Vec<f64>,pub partition:RowPartition,pub iterations:usize,
    pub residual_norm:f64,pub rhs_norm:f64,pub status:IterativeStatus,pub collective_calls:u64,
}
struct Exchange<'a,C:?Sized>{comm:&'a C,phase:u64}
impl<C:SolverCommunicator+?Sized>Exchange<'_,C>{
    fn gather(&mut self,x:&[f64])->Result<Vec<Vec<f64>>,SolverError>{
        self.phase=self.phase.checked_add(1).ok_or(SolverError::SizeOverflow)?;
        let all=self.comm.all_gather_fixed(self.phase,x)?;
        if all.len()!=self.comm.size()||all.iter().any(|v|v.len()!=x.len()){return Err(SolverError::Communication("all-gather shape mismatch".into()));}
        for v in &all{finite(v)?;}Ok(all)
    }
    fn sum(&mut self,x:f64)->Result<f64,SolverError>{checked(x,"local reduction")?;sum_iter(self.gather(&[x])?.iter().map(|v|v[0]))}
    fn norm(&mut self,x:&[f64])->Result<f64,SolverError>{
        let local=x.iter().fold(0.0f64,|a,b|a.max(b.abs()));
        let scale=self.gather(&[local])?.iter().fold(0.0f64,|a,b|a.max(b[0]));
        if scale==0.0{return Ok(0.0);}let s=sum_iter(x.iter().map(|&x|(x/scale)*(x/scale)))?;
        checked(scale*self.sum(s)?.sqrt(),"distributed norm")
    }
    fn vector(&mut self,local:&[f64],n:usize)->Result<Vec<f64>,SolverError>{
        let width=n.div_ceil(self.comm.size());let mut padded=zeros(width)?;padded[..local.len()].copy_from_slice(local);
        let all=self.gather(&padded)?;let mut full=zeros(n)?;
        for(rank,v)in all.iter().enumerate(){let p=RowPartition::balanced(n,rank,self.comm.size())?;full[p.start..p.start+p.rows].copy_from_slice(&v[..p.rows]);}Ok(full)
    }
}
/// Solve SPD A x=b with balanced contiguous row partitioning. optional Jacobi is
/// local diagonal scaling. No checkpoint/restart inside a solve: an error aborts
/// the communicator; create a fresh communicator before retrying.
pub fn distributed_cg<C:SolverCommunicator+?Sized>(comm:&C,a:&DistributedCsr<'_>,b:&[f64],
initial:Option<&[f64]>,jacobi:bool,options:CgOptions)->Result<DistributedCgReport,SolverError>{
    let result=solve(comm,a,b,initial,jacobi,options);
    if let Err(ref e)=result{comm.abort(&e.to_string());}result
}
fn solve<C:SolverCommunicator+?Sized>(comm:&C,a:&DistributedCsr<'_>,b:&[f64],initial:Option<&[f64]>,jacobi:bool,options:CgOptions)->Result<DistributedCgReport,SolverError>{
    let partition=RowPartition::balanced(a.partition.global_rows,comm.rank(),comm.size())?;
    if partition!=a.partition||b.len()!=partition.rows||initial.is_some_and(|x|x.len()!=b.len()){return Err(SolverError::Shape("distributed local partition/RHS"));}
    finite(b)?;options.tolerance.validate()?;
    if options.residual_recompute_interval==0||options.residual_recompute_interval>u32::MAX as usize||options.max_iterations>u32::MAX as usize||(options.tolerance.absolute==0.0&&options.tolerance.relative==0.0){return Err(SolverError::InvalidOption("distributed CG options"));}
    let mut net=Exchange{comm,phase:0};
    let config=[partition.global_rows as f64,options.max_iterations as f64,options.residual_recompute_interval as f64,
        options.tolerance.absolute,options.tolerance.relative,if jacobi{1.0}else{0.0},if initial.is_some(){1.0}else{0.0}];
    if net.gather(&config)?.iter().any(|v|v.as_slice()!=config){return Err(SolverError::Communication("ranks disagree on solve options".into()));}
    let diag=if jacobi{a.diagonal()?}else{vec![1.0;b.len()]};
    let mut x=initial.map_or_else(||vec![0.0;b.len()],|v|v.to_vec());finite(&x)?;
    let n=partition.global_rows;let rhs_norm=net.norm(b)?;let target=options.tolerance.threshold(rhs_norm)?;
    let ax=a.apply(&net.vector(&x,n)?)?;let mut r:Vec<f64>=b.iter().zip(&ax).map(|(&b,&ax)|b-ax).collect();finite(&r)?;
    let mut residual=net.norm(&r)?;
    let mut iterations=0;let mut status=IterativeStatus::MaxIterations;
    let mut z=zeros(b.len())?;for i in 0..z.len(){z[i]=checked(r[i]/diag[i],"distributed preconditioner")?;}
    let mut p=z.clone();let mut rho=net.sum(dot(&r,&z)?)?;
    if residual<=target{status=IterativeStatus::Converged;}
    else{for it in 1..=options.max_iterations{
        if rho<=0.0{return Err(SolverError::Breakdown("distributed nonpositive residual/preconditioner"));}
        let ap=a.apply(&net.vector(&p,n)?)?;let curvature=net.sum(dot(&p,&ap)?)?;
        if curvature<=0.0{return Err(SolverError::Breakdown("distributed CG requires SPD operator"));}
        let alpha=checked(rho/curvature,"distributed alpha")?;
        for i in 0..x.len(){x[i]=checked(alpha.mul_add(p[i],x[i]),"distributed iterate")?;r[i]=checked((-alpha).mul_add(ap[i],r[i]),"distributed residual")?;}
        residual=net.norm(&r)?;let replace=it%options.residual_recompute_interval==0||residual<=target||it==options.max_iterations;
        if replace{let ax=a.apply(&net.vector(&x,n)?)?;for i in 0..r.len(){r[i]=checked(b[i]-ax[i],"distributed true residual")?;}residual=net.norm(&r)?;}
        iterations=it;if residual<=target{status=IterativeStatus::Converged;break;}
        if it==options.max_iterations{break;}
        for i in 0..z.len(){z[i]=checked(r[i]/diag[i],"distributed Jacobi")?;}
        let next=net.sum(dot(&r,&z)?)?;if next<=0.0{return Err(SolverError::Breakdown("distributed CG rho underflow"));}
        let beta=checked(next/rho,"distributed beta")?;
        for i in 0..p.len(){p[i]=if replace{z[i]}else{checked(beta.mul_add(p[i],z[i]),"distributed direction")?};}rho=next;
    }}
    Ok(DistributedCgReport{local_solution:x,partition,iterations,residual_norm:residual,rhs_norm,status,collective_calls:net.phase})
}

/// Reuses the EXISTING GXCL transport (CPU/TCP path), not a second socket stack.
/// A dedicated session is required; don't interleave unrelated collectives.
#[cfg(feature="collective")]
pub struct RucclCommunicator<'a>{session:&'a dyn ruccl::rank::RankTransport}
#[cfg(feature="collective")]
impl<'a>RucclCommunicator<'a>{
    pub fn new(session:&'a dyn ruccl::rank::RankTransport)->Result<Self,SolverError>{
        use ruccl::rank::CollectiveTransport;
        if !matches!(session.transport(),CollectiveTransport::TcpHostStaged|CollectiveTransport::TcpPeer){return Err(SolverError::InvalidOption("solver adapter currently supports GXCL TCP transports only"));}
        Ok(Self{session})
    }
}
#[cfg(feature="collective")]
impl SolverCommunicator for RucclCommunicator<'_>{
    fn rank(&self)->usize{self.session.rank()as usize}fn size(&self)->usize{self.session.world_size()as usize}
    fn all_gather_fixed(&self,phase:u64,local:&[f64])->Result<Vec<Vec<f64>>,SolverError>{
        use ruccl::rank::{Opcode,ElementType,ANY_RANK};
        let mut words=Vec::with_capacity(local.len()+2);words.push((phase>>32)as f64);words.push((phase as u32)as f64);words.extend_from_slice(local);
        let bytes:Vec<u8>=words.iter().flat_map(|x|x.to_le_bytes()).collect();
        let response=self.session.exchange(Opcode::AllGather,ElementType::F64,ANY_RANK,words.len()as u64,bytes)
            .map_err(|e|SolverError::Communication(e.to_string()))?;
        let width=words.len().checked_mul(8).ok_or(SolverError::SizeOverflow)?;
        if response.payload.len()!=width.checked_mul(self.size()).ok_or(SolverError::SizeOverflow)?{return Err(SolverError::Communication("GXCL allgather byte count".into()));}
        let mut out=Vec::new();for rank_bytes in response.payload.chunks_exact(width){
            let mut v=Vec::with_capacity(words.len());for b in rank_bytes.chunks_exact(8){let mut a=[0;8];a.copy_from_slice(b);v.push(f64::from_le_bytes(a));}
            if v[..2]!=words[..2]{return Err(SolverError::Communication("solver collective phase mismatch".into()));}out.push(v[2..].to_vec());
        }Ok(out)
    }
    fn abort(&self,reason:&str){let _=self.session.abort(reason);}
}
