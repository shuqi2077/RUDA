// SPDX-License-Identifier: Apache-2.0
use crate::{distributed::*,SolverError,CgOptions,IterativeStatus};
use std::{sync::mpsc,thread,time::Duration};
enum Message{Call{rank:usize,phase:u64,data:Vec<f64>,reply:mpsc::Sender<Result<Vec<Vec<f64>>,SolverError>>},Abort(String)}
struct Comm{rank:usize,size:usize,sender:mpsc::Sender<Message>}
impl SolverCommunicator for Comm{
    fn rank(&self)->usize{self.rank}fn size(&self)->usize{self.size}
    fn all_gather_fixed(&self,phase:u64,x:&[f64])->Result<Vec<Vec<f64>>,SolverError>{let(tx,rx)=mpsc::channel();self.sender.send(Message::Call{rank:self.rank,phase,data:x.to_vec(),reply:tx}).map_err(|_|SolverError::Communication("closed group".into()))?;
        rx.recv_timeout(Duration::from_secs(3)).map_err(|_|SolverError::Communication("test timeout".into()))?}
    fn abort(&self,reason:&str){let _=self.sender.send(Message::Abort(reason.to_owned()));}
}
fn group(size:usize)->(Vec<Comm>,thread::JoinHandle<()>){let(tx,rx)=mpsc::channel();let comms=(0..size).map(|rank|Comm{rank,size,sender:tx.clone()}).collect();drop(tx);
    let handle=thread::spawn(move||{let mut slots:Vec<Option<(u64,Vec<f64>,mpsc::Sender<Result<Vec<Vec<f64>>,SolverError>>)>>=(0..size).map(|_|None).collect();let mut failed:Option<String>=None;
        while let Ok(msg)=rx.recv_timeout(Duration::from_secs(4)){
            match msg{Message::Abort(s)=>{failed=Some(s.clone());for slot in &mut slots{if let Some((_,_,reply))=slot.take(){let _=reply.send(Err(SolverError::Communication(s.clone())));}}}
                Message::Call{rank,phase,data,reply}=>{
                    if let Some(ref s)=failed{let _=reply.send(Err(SolverError::Communication(s.clone())));continue;}
                    assert!(slots[rank].is_none());slots[rank]=Some((phase,data,reply));
                    if slots.iter().all(Option::is_some){let expected=slots[0].as_ref().unwrap();let ok=slots.iter().all(|s|{let s=s.as_ref().unwrap();s.0==expected.0&&s.1.len()==expected.1.len()});
                        let all:Vec<Vec<f64>>=slots.iter().map(|s|s.as_ref().unwrap().1.clone()).collect();for slot in &mut slots{let(_,_,reply)=slot.take().unwrap();let _=reply.send(if ok{Ok(all.clone())}else{Err(SolverError::Communication("phase mismatch".into()))});}}
                }
            }
        }
    });(comms,handle)
}
fn local_matrix(n:usize,p:RowPartition)->(Vec<usize>,Vec<usize>,Vec<f64>,Vec<f64>){let(mut offsets,mut cols,mut values,mut b)=(vec![0],vec![],vec![],vec![]);
    for i in p.start..p.start+p.rows{let mut rhs=3.;if i>0{cols.push(i-1);values.push(-1.);rhs-=1.;}cols.push(i);values.push(3.);if i+1<n{cols.push(i+1);values.push(-1.);rhs-=1.;}offsets.push(values.len());b.push(rhs);}(offsets,cols,values,b)}
#[test]fn row_partitioned_cg_uneven_and_empty_ranks(){for(n,size)in[(11,1),(11,2),(11,3),(2,5)]{let(comms,server)=group(size);let workers:Vec<_>=comms.into_iter().map(|c|thread::spawn(move||{
    let p=RowPartition::balanced(n,c.rank(),c.size()).unwrap();let(o,j,v,b)=local_matrix(n,p);let a=DistributedCsr::new(p,&o,&j,&v).unwrap();distributed_cg(&c,&a,&b,None,true,Default::default()).unwrap()
})).collect();let reports:Vec<_>=workers.into_iter().map(|w|w.join().unwrap()).collect();for r in &reports{assert_eq!(r.status,IterativeStatus::Converged);assert!(r.local_solution.iter().all(|x|(x-1.).abs()<1e-8));assert_eq!(r.iterations,reports[0].iterations);}server.join().unwrap();}}
#[test]fn distributed_options_mismatch_fails_all_ranks(){let(comms,server)=group(2);let workers:Vec<_>=comms.into_iter().map(|c|thread::spawn(move||{let p=RowPartition::balanced(4,c.rank(),2).unwrap();let(o,j,v,b)=local_matrix(4,p);let a=DistributedCsr::new(p,&o,&j,&v).unwrap();let opt=CgOptions{max_iterations:10+c.rank(),..Default::default()};assert!(distributed_cg(&c,&a,&b,None,true,opt).is_err());})).collect();for w in workers{w.join().unwrap();}server.join().unwrap();}
#[test]fn distributed_local_error_aborts_peers(){let(comms,server)=group(2);let workers:Vec<_>=comms.into_iter().map(|c|thread::spawn(move||{let p=RowPartition::balanced(4,c.rank(),2).unwrap();let(o,j,v,mut b)=local_matrix(4,p);if c.rank()==1{b[0]=f64::NAN;}let a=DistributedCsr::new(p,&o,&j,&v).unwrap();assert!(distributed_cg(&c,&a,&b,None,true,Default::default()).is_err());})).collect();for w in workers{w.join().unwrap();}server.join().unwrap();}
#[test]fn local_communicator_zero_rhs(){let a=DistributedCsr::new(RowPartition::balanced(2,0,1).unwrap(),&[0,1,2],&[0,1],&[2.,3.]).unwrap();let r=distributed_cg(&LocalCommunicator,&a,&[0.,0.],None,true,Default::default()).unwrap();assert_eq!(r.iterations,0);assert_eq!(r.status,IterativeStatus::Converged);}
#[test]fn partition_rejects_invalid_rank(){assert!(RowPartition::balanced(10,2,2).is_err());}
