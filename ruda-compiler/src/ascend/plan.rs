//! Compiler-local UB allocation. Not a second public kernel IR.
use super::{Result,invalid,lower::{Program,Node}};
#[derive(Debug)]
pub(super) struct Allocation { pub node_slots:Vec<Option<usize>>,pub slots:usize }
/// Never overwrite an operand during the instruction using it. Outputs survive
/// until copy-out. Vector barriers are emitted before a slot can be reused.
pub(super) fn allocate(p:&Program,reuse:bool)->Result<Allocation>{
    let n=p.nodes.len();let mut last:Vec<usize>=(0..n).collect();
    for (i,node) in p.nodes.iter().enumerate(){for a in node.inputs(){if a>=i{return Err(invalid("invalid SSA node order"));}last[a]=last[a].max(i);}}
    for &(_,v) in &p.stores{if v>=n{return Err(invalid("invalid store node"));}last[v]=n;}
    let mut ends=Vec::<usize>::new();let mut node_slots=Vec::new();
    for (i,node) in p.nodes.iter().enumerate(){
        if matches!(node,Node::Input(_)){node_slots.push(None);continue;}
        let free=if reuse {ends.iter().position(|&end|end<i)}else{None};
        let slot=free.unwrap_or(ends.len());if slot==ends.len(){ends.push(last[i]);}else{ends[slot]=last[i];}
        node_slots.push(Some(slot));
    }
    Ok(Allocation{node_slots,slots:ends.len()})
}
