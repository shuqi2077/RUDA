// SPDX-License-Identifier: Apache-2.0
//! Multiprocess CPU/TCP demo using existing ruCCL. NOT a GPU distributed solver.
//! Terminal 1: distributed-cg server 127.0.0.1:45670 2
//! Terminal 2: distributed-cg rank   127.0.0.1:45670 2 0 32
//! Terminal 3: distributed-cg rank   127.0.0.1:45670 2 1 32
//! Fixed demonstration identity: don't expose this server to an untrusted network.
use rusolver::distributed::*;
use ruccl::rank::{TcpRendezvousServer,TcpRankSession,UniqueId,CollectiveTransport};
use std::{io::{self,Write},time::Duration};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let args:Vec<String>=std::env::args().collect();
    if args.len()<4{return Err("usage: distributed-cg server|rank address world [rank n]".into());}
    let address=&args[2];let world:usize=args[3].parse()?;let id=UniqueId::from_bytes([0x52;16]);
    if args[1]=="server"{
        let server=TcpRendezvousServer::bind(address,id,world)?.with_p2p_rails(1)?.with_transport(CollectiveTransport::TcpHostStaged)?
            .with_collective_timeout(Duration::from_secs(30))?;
        println!("LISTEN {}",server.local_addr()?);io::stdout().flush()?;server.run()?;return Ok(());
    }
    if args[1]!="rank"||args.len()!=6{return Err("rank mode needs rank and global order".into());}
    let rank:usize=args[4].parse()?;let n:usize=args[5].parse()?;let p=RowPartition::balanced(n,rank,world)?;
    let(mut offsets,mut columns,mut values,mut rhs)=(vec![0],Vec::new(),Vec::new(),Vec::new());
    for i in p.start..p.start+p.rows{let mut b=3.;if i>0{columns.push(i-1);values.push(-1.);b-=1.;}
        columns.push(i);values.push(3.);if i+1<n{columns.push(i+1);values.push(-1.);b-=1.;}offsets.push(values.len());rhs.push(b);}
    let a=DistributedCsr::new(p,&offsets,&columns,&values)?;
    let session=TcpRankSession::connect_with_transport(address,id,rank as u32,world as u32,Duration::from_secs(30),1,CollectiveTransport::TcpHostStaged)?;
    let comm=RucclCommunicator::new(&session)?;let result=distributed_cg(&comm,&a,&rhs,None,true,Default::default())?;
    if result.status!=rusolver::IterativeStatus::Converged||result.local_solution.iter().any(|x|(x-1.).abs()>1e-8){return Err("distributed result check failed".into());}
    println!("PASS rank={rank} rows={} iterations={} residual={:.4e} collectives={}",p.rows,result.iterations,result.residual_norm,result.collective_calls);Ok(())
}
