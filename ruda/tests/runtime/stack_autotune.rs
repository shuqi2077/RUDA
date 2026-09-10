//! Exercises the real LocalTuner adapter, allocation isolation, validation and queue completion.
//! The dummy backend is a host implementation, not a simulated GPU performance claim.
#![allow(unsafe_code)]
mod dummy;
use dummy::*;
use ruda::runtime::{server::{Handle, RudaCount, KernelArguments}, tune::{AutotuneOutput, LocalTuner, Tunable, TunableSet, TuneGroup, stack::*}};
use std::sync::Arc;

#[derive(Clone)]
struct Checked { client: DummyClient, out: Handle }
impl AutotuneOutput for Checked {
    fn validate_for_tuning(&self, other:&Self,_:f64,_:f64,max:u64)->Result<bool,String>{
        if self.out.size_in_used()+other.out.size_in_used()>max {return Ok(false);}
        let a=self.client.read_one(self.out.clone()).map_err(|e|format!("{e:?}"))?;
        let b=other.client.read_one(other.out.clone()).map_err(|e|format!("{e:?}"))?;
        if a==b {Ok(true)} else {Err("wrong output".into())}
    }
    #[cfg(feature="runtime-autotune-checks")]
    fn check_equivalence(&self,other:Self){assert!(self.validate_for_tuning(&other,0.,0.,1024).unwrap());}
}
fn run(client:&DummyClient, handles:Vec<Handle>, wrong:bool)->Result<Checked,String>{
    let out=handles[2].clone();
    if wrong {
        client.launch(Box::new(KernelTask::new(DummyElementwiseAdditionSlowWrong)), RudaCount::Static(1,1,1),
            KernelArguments::new().with_buffers(handles.into_iter().map(Handle::binding).collect()));
    } else {
        client.launch(Box::new(KernelTask::new(DummyElementwiseAddition)), RudaCount::Static(1,1,1),
            KernelArguments::new().with_buffers(handles.into_iter().map(Handle::binding).collect()));
    }
    Ok(Checked {client:client.clone(),out})
}
#[test]
fn stack_adapter_checks_all_groups_and_preserves_real_output() {
    let ctl=enable_stack_autotune(StackPolicy {warmups:1,samples:3,..StackPolicy::default()},None).unwrap();
    let client=test_client(&DummyDevice);
    let (a,b,c)=(client.create_from_slice(&[0,1,2]),client.create_from_slice(&[4,4,4]),client.create_from_slice(&[99,99,99]));
    let prep=client.clone(); let correct=client.clone(); let wrong=client.clone(); let later=client.clone();
    let high=TuneGroup::<String>::new("high", |_|2); let low=TuneGroup::<String>::new("low", |_|1);
    let set:TunableSet<String,Vec<Handle>,Checked>=TunableSet::new(
        |_input:&Vec<Handle>|"shape3".into(), move |_key:&String,input:&Vec<Handle>| {
            vec![input[0].clone(),input[1].clone(),prep.empty(usize::try_from(input[2].size_in_used()).unwrap())]
        })
        .with(Tunable::new("ref",move |input|run(&correct,input,false)).group(&high, |_|1))
        .with(Tunable::new("wrong",move |input|run(&wrong,input,true)).group(&high, |_|0))
        .with(Tunable::new("late-correct",move |input|run(&later,input,false)).group(&low, |_|0))
        .with_stack_tuning(0,"adapter-test-v1",|input|format!("bytes={}",input[2].size_in_used()));
    let set=Arc::new(set); let local=LocalTuner::<String,String>::new("stack-adapter-host-test");
    let out=local.execute(&"dummy".into(),&client,set.clone(),vec![a.clone(),b.clone(),c.clone()]);
    assert_eq!(client.read_one(out.out).unwrap().to_vec(),vec![4,5,6]);
    let reports=ctl.reports(); let report=reports.iter().find(|r|r.operation.contains("stack-adapter-host-test")).unwrap();
    assert_eq!(report.candidates.len(),3); assert!(report.candidates.iter().any(|r|r.name=="wrong" && !r.verified));
    assert!(report.candidates.iter().any(|r|r.name=="late-correct" && r.verified));
    let before=ctl.stats(); local.execute(&"dummy".into(),&client,set,vec![a,b,c]);
    assert_eq!(ctl.stats().tunes,before.tunes); assert!(ctl.stats().memory_hits>before.memory_hits);
}
