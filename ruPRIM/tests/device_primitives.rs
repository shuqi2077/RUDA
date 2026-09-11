#![cfg(feature = "cuda-tests")]
#![allow(unsafe_code)]
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, transfer::from_data, readback::into_data_sync};
use ruda_core::tensor::data::TensorData;
use ruda_test_runtime::TestRuntime as R;
use ruprim::collective::*;
use ruprim::device::{self, select::{RudaPredicate, RudaPredicateExpand}};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruprim::collective::record::*;
use device::record::{RudaRecordBuffer, RudaRecordBytes};

fn tensor<T: TensorElement>(data: &[T]) -> RudaTensor<R> {
    from_data(TensorData::new(data.to_vec(), [data.len()]), &Default::default())
}
fn read<T: TensorElement>(value: RudaTensor<R>) -> Vec<T> { into_data_sync(value).to_vec::<T>().unwrap() }
#[derive(RudaType, RudaLaunch)]
struct Even;
#[ruda]
impl RudaPredicate<i32> for Even { fn test(&self, value: i32) -> bool { value % 2 == 0 } }

#[test]
fn scan_reduce_sizes_and_in_place() {
    for n in [0, 1, 31, 32, 33, 257, 1025] {
        let data: Vec<i32> = (0..n).map(|i| i % 11 - 5).collect();
        let input = tensor(&data);
        let expected: Vec<_> = data.iter().scan(0, |sum, x| { *sum += x; Some(*sum) }).collect();
        assert_eq!(read::<i32>(device::scan::inclusive_scan::<R,i32,RudaSum>(&input, RudaSumLaunch::new(),32).unwrap()), expected);
        assert_eq!(read::<i64>(device::typed::reduce::<R,i32,i64,RudaSum>(&input,7,RudaSumLaunch::new(),32).unwrap()), [7+data.iter().map(|v|*v as i64).sum::<i64>()]);
        device::scan::scan_into::<R,i32,RudaSum>(&input,&input,Some(7),true,RudaSumLaunch::new(),32).unwrap();
        let expected: Vec<_> = data.iter().scan(7, |sum,x| { let old=*sum; *sum+=x; Some(old) }).collect();
        assert_eq!(read::<i32>(input),expected);
    }
}

#[test]
fn sorts_topk_and_double_buffers() {
    let data=[3,-2,3,i32::MIN,0,i32::MAX,-2];
    let input=tensor(&data);
    let mut expected=data.to_vec(); expected.sort();
    assert_eq!(read::<i32>(device::sort::merge_sort_keys::<R,i32,RudaAscending>(&input,RudaAscendingLaunch::new()).unwrap()),expected);
    assert_eq!(read::<i32>(device::radix::sort_keys::<R,i32>(&input,0,32,false,32).unwrap()),expected);
    let mut buffers=device::double_buffer::RudaDoubleBuffer::new(tensor(&[0i32;7]),input.clone(),1).unwrap();
    device::double_buffer::radix_keys::<R,i32>(&mut buffers,0,32,false,32).unwrap();
    assert_eq!(read::<i32>(buffers.current().clone()),expected);
    assert_eq!(buffers.selector(),1);
    for k in [0,1,3,7,9] {
        let mut actual=read::<i32>(device::topk::keys::<R,i32>(&input,k,true,32).unwrap()); actual.sort();
        assert_eq!(actual,expected[7-k.min(7)..]);
    }
}

#[test]
fn selection_partition_and_runs() {
    let input=tensor(&[3,2,2,5,4,4,4]);
    let selected=device::select::select_if::<R,i32,Even>(&input,EvenLaunch::new(),32,false).unwrap();
    let count=read::<u64>(selected.count)[0] as usize;
    assert_eq!(&read::<i32>(selected.values)[..count], [2,2,4,4,4]);
    let partition=device::partition::two_way::<R,i32,Even>(&input,EvenLaunch::new(),32).unwrap();
    assert_eq!(read::<u64>(partition.counts),[5,2]);
    assert_eq!(&read::<i32>(partition.rejected)[..2],[3,5]);
    let runs=device::run_length::encode::<R,i32,RudaEqual>(&input,RudaEqualLaunch::new(),32).unwrap();
    assert_eq!(read::<u64>(runs.count.clone()),[4]);
    assert_eq!(&read::<u64>(runs.lengths.clone())[..4],[1,2,1,3]);
    assert_eq!(&read::<u32>(device::typed::indices_prefix::<R,u64,u32>(&runs.lengths,&runs.count).unwrap())[..4],[1,2,1,3]);
}

#[test]
fn histogram_mixed_types_channels_strides_and_limits() {
    use device::histogram::{self,RudaHistogramRegion as Region};
    let input=tensor(&[0.5f32,10.,1.5,11.,999.,999.,2.5,12.,3.5,13.]);
    let region=Region{width:2,rows:2,channels:2,row_stride:6};
    let result=histogram::even_mixed::<R,f32,i32,u32>(&input,region,&[(8,0,4),(4,10,14)]).unwrap();
    assert_eq!(read::<u32>(result[0].clone()),[0,1,0,1,0,1,0,1]);
    assert_eq!(read::<u32>(result[1].clone()),[1,1,1,1]);
    let result=histogram::range_mixed::<R,f32,i32,u32>(&input,region,&[tensor(&[0,1,2,3,4])]).unwrap();
    assert_eq!(read::<u32>(result[0].clone()),[1,1,1,1]);
    let extremes=tensor(&[i64::MIN,-1,0,i64::MAX-1,i64::MAX]);
    let region=Region{width:5,rows:1,channels:1,row_stride:5};
    let bins=histogram::even::<R,i64,u32>(&extremes,region,&[(2,i64::MIN,i64::MAX)]).unwrap();
    assert_eq!(read::<u32>(bins[0].clone()),[2,2]);
}

ruprim::ruda_record! { struct Pair { key:i32, tag:u32 } }
ruprim::ruda_decomposer!(PairBits, PairBitsLaunch for Pair { key:i32 });
ruprim::ruda_decomposer!(PairFullBits, PairFullBitsLaunch for Pair { key:i32, tag:u32 });
#[derive(RudaType,RudaLaunch)]
struct KeyOrder;
impl<Rt:Runtime> Clone for KeyOrderLaunch<Rt> { fn clone(&self)->Self { Self::new() } }
#[ruda]
impl RudaCompare<Pair> for KeyOrder { fn before(&self,a:Pair,b:Pair)->bool { a.key<b.key } }
#[derive(RudaType,RudaLaunch)]
struct KeyEqual;
#[ruda]
impl RudaKeyEqual<Pair> for KeyEqual { fn equal(&self,a:Pair,b:Pair)->bool { a.key==b.key } }
#[derive(RudaType,RudaLaunch)]
struct RecordEven;
#[ruda]
impl RudaPredicate<Pair> for RecordEven { fn test(&self,a:Pair)->bool { a.key%2==0 } }

#[ruda(launch_unchecked,address_type="u64")]
fn upload(input:&LinearView<i32>, output:&mut RudaRecordBytes) {
    let i=ABSOLUTE_POS;
    if i<input.shape() { <RudaRecordBytes as RudaWrite<Pair>>::write(output,i,Pair{key:input[i],tag:i as u32}); }
}
#[ruda(launch_unchecked,address_type="u64")]
fn download(input:&RudaRecordBytes,keys:&mut LinearView<i32,ReadWrite>,tags:&mut LinearView<u32,ReadWrite>,count:usize) {
    let i=ABSOLUTE_POS;
    if i<count { let v=<RudaRecordBytes as RudaRead<Pair>>::read(input,i); keys[i]=v.key; tags[i]=v.tag; }
}
fn records(data:&[i32])->RudaRecordBuffer<R,Pair> {
    let input=tensor(data); let out=RudaRecordBuffer::allocate(&input,data.len()).unwrap();
    if !data.is_empty() { unsafe { upload::launch_unchecked::<R>(&input.client,RudaCount::Static(data.len().div_ceil(32) as u32,1,1),RudaDim::new_1d(32),input.clone().into_linear_view(),out.view()); } }
    out
}
fn read_records(input:&RudaRecordBuffer<R,Pair>,count:usize)->(Vec<i32>,Vec<u32>) {
    let keys=tensor(&vec![0i32;count]); let tags=tensor(&vec![0u32;count]);
    if count>0 { unsafe { download::launch_unchecked::<R>(input.client(),RudaCount::Static(count.div_ceil(32) as u32,1,1),RudaDim::new_1d(32),input.view(),keys.clone().into_linear_view(),tags.clone().into_linear_view(),count); } }
    (read(keys),read(tags))
}

#[test]
fn record_merge_sort_selection_and_partition() {
    use device::record;
    let input=records(&[3,2,2,5,4,4,4]);
    let sorted=record::merge_sort_keys::<R,Pair,KeyOrder>(&input,KeyOrderLaunch::new()).unwrap();
    assert_eq!(read_records(&sorted,7),(vec![2,2,3,4,4,4,5],vec![1,2,0,4,5,6,3]));
    let unique=record::select::unique::<R,Pair,KeyEqual>(&input,KeyEqualLaunch::new(),32).unwrap();
    assert_eq!(read::<u64>(unique.count),[4]);
    assert_eq!(read_records(&unique.values,4).0,[3,2,5,4]);
    let partition=record::partition::two_way::<R,Pair,RecordEven>(&input,RecordEvenLaunch::new(),32).unwrap();
    assert_eq!(read::<u64>(partition.counts),[5,2]);
    assert_eq!(read_records(&partition.rejected,2).0,[3,5]);
    let left=records(&[1,2,2]); let right=records(&[2,3]);
    let merged=record::merge::keys::<R,Pair,KeyOrder>(&left,&right,KeyOrderLaunch::new()).unwrap();
    assert_eq!(read_records(&merged,5),(vec![1,2,2,2,3],vec![0,1,2,0,1]));
    let bounds=record::merge::bounds::<R,Pair,KeyOrder>(&merged,&records(&[0,2,4]),KeyOrderLaunch::new(),false).unwrap();
    assert_eq!(read::<u64>(bounds),[0,1,5]);
}

#[test]
fn record_decomposed_double_buffer_and_segment_gaps() {
    use device::record::double_buffer::*;
    let mut keys=RudaRecordDoubleBuffer::new(records(&[9,3,-2,3,8,2,1]),records(&[0;7]),0).unwrap();
    segmented_radix_keys::<R,Pair,PairBits>(&mut keys,&tensor(&[1u64,5]),&tensor(&[4u64,7]),PairBitsLaunch::new(),0,32,false,32).unwrap();
    assert_eq!(read_records(keys.current(),7).0,[9,-2,3,3,8,1,2]);
    assert_eq!(keys.selector(),0);
    radix_keys::<R,Pair,PairFullBits>(&mut keys,PairFullBitsLaunch::new(),0,64,false,32).unwrap();
    assert_eq!(read_records(keys.current(),7),(vec![-2,1,2,3,3,8,9],vec![2,6,5,1,3,4,0]));
}

#[ruda(launch_unchecked)]
fn warp_block_collectives(output:&mut Array<u32>) {
    let value=UNIT_POS+1;
    let sum=RudaSum{};
    let prefix=ruprim::warp::inclusive_scan::<u32,RudaSum>(value,&sum,16u32,16u32);
    output[UNIT_POS as usize]=prefix;
    let mut scratch=SharedMemory::<u32>::new(32usize);
    let rotated=ruprim::block::shuffle::rotate_access::<u32,SharedMemory<u32>>(value,&mut scratch,1u32,32u32);
    output[32usize+UNIT_POS as usize]=rotated;
}

#[test]
fn logical_warps_and_block_shuffle() {
    let out=tensor(&[0u32;64]);
    unsafe { warp_block_collectives::launch_unchecked::<R>(&out.client,RudaCount::Static(1,1,1),RudaDim::new_1d(32),ArrayArg::from_raw_parts(out.handle.clone(),64)); }
    let values=read::<u32>(out);
    for lane in 0..32 {
        let base=lane/16*16;
        assert_eq!(values[lane],((base+1)..=(lane+1)).sum::<usize>() as u32);
        assert_eq!(values[32+lane],((lane+1)%32+1) as u32);
    }
}

#[derive(RudaType, RudaLaunch)]
struct PairSum;
impl<Rt: Runtime> Clone for PairSumLaunch<Rt> { fn clone(&self) -> Self { Self::new() } }
#[ruda]
impl RudaBinaryOp<Pair> for PairSum {
    fn combine(&self, a: Pair, b: Pair) -> Pair { Pair { key: a.key + b.key, tag: a.tag + b.tag } }
}

#[test]
fn record_scan_reduce_and_run_lengths() {
    use device::record;
    for n in [0usize, 1, 31, 33, 1025] {
        let data: Vec<i32> = (0..n).map(|i| i as i32 % 7 - 3).collect();
        let input = records(&data);
        let initial = records(&[7]);
        let output = record::inclusive_scan::<R,Pair,PairSum>(&input,PairSumLaunch::new(),32).unwrap();
        let expected: Vec<_> = data.iter().scan(0, |sum,x| { *sum+=x; Some(*sum) }).collect();
        let tags: Vec<_> = (0..n as u32).scan(0, |sum,x| { *sum+=x; Some(*sum) }).collect();
        assert_eq!(read_records(&output,n),(expected,tags));
        let output = record::scan_init::<R,Pair,PairSum>(&input,&initial,PairSumLaunch::new(),true,32).unwrap();
        let expected: Vec<_> = data.iter().scan(7, |sum,x| { let old=*sum; *sum+=x; Some(old) }).collect();
        assert_eq!(read_records(&output,n).0,expected);
        let output = record::reduce::<R,Pair,PairSum>(&input,&initial,PairSumLaunch::new(),32).unwrap();
        assert_eq!(read_records(&output,1),(vec![7+data.iter().sum::<i32>()],vec![(0..n as u32).sum()]));
    }
    let runs = record::run_length::nontrivial::<R,Pair,KeyEqual>(&records(&[3,2,2,5,4,4,4]),KeyEqualLaunch::new(),32).unwrap();
    assert_eq!(read::<u64>(runs.count),[2]);
    assert_eq!(&read::<u64>(runs.lengths)[..2],[2,3]);
    assert_eq!(&read::<u64>(runs.offsets)[..2],[1,4]);
}

#[test]
fn independent_segments_copy_and_adjacent() {
    let data: Vec<i32> = (0..100).map(|i| i % 9 - 4).collect();
    let input = tensor(&data);
    let begins = tensor(&[1u64,5,80]); let ends = tensor(&[1u64,72,99]);
    assert_eq!(read::<i32>(device::segments::reduce::<R,i32,RudaSum>(&input,&begins,&ends,7,RudaSumLaunch::new(),32).unwrap()),
        [7,7+data[5..72].iter().sum::<i32>(),7+data[80..99].iter().sum::<i32>()]);
    let output = tensor(&[99i32;100]);
    device::segments::scan_into::<R,i32,RudaSum>(&input,&begins,&ends,&begins,&output,Some(7),true,RudaSumLaunch::new(),32).unwrap();
    let mut expected = vec![99i32;100];
    for (begin,end) in [(5,72),(80,99)] {
        let mut sum=7;
        for i in begin..end { expected[i]=sum; sum+=data[i]; }
    }
    assert_eq!(read::<i32>(output),expected);
    let copied=tensor(&[99i32;100]);
    device::copy::ranges::<R,i32>(&input,&copied,&begins,&begins,&tensor(&[0u64,67,19]),32).unwrap();
    let mut expected=vec![99i32;100]; expected[5..72].copy_from_slice(&data[5..72]); expected[80..99].copy_from_slice(&data[80..99]);
    assert_eq!(read::<i32>(copied),expected);
    let output=device::adjacent::difference::<R,i32,RudaSubtract>(&input,RudaSubtractLaunch::new(),false,true).unwrap();
    let mut expected=data.clone(); for i in 1..data.len() { expected[i]=data[i]-data[i-1]; }
    assert_eq!(read::<i32>(output),expected);
}

#[ruda(launch_unchecked, explicit_define)]
fn conversion_kernel<T: Numeric>(input:&Array<f64>,output:&mut Array<T>, count:usize) {
    if ABSOLUTE_POS<count { output[ABSOLUTE_POS]=T::cast_from(input[ABSOLUTE_POS]); }
}
fn conversions<T: TensorElement>(input:&RudaTensor<R>,expected:&[T]) {
    let output=ruda_kernel::tensor::allocation::empty_device_dtype(input.client.clone(),input.device.clone(),
        ruda_core::tensor::Shape::new([expected.len()]),<T as ruda_core::tensor::element::Element>::dtype());
    unsafe { conversion_kernel::launch_unchecked::<T,R>(&input.client,RudaCount::Static(1,1,1),RudaDim::new_1d(32),
        ArrayArg::from_raw_parts(input.handle.clone(),expected.len()),ArrayArg::from_raw_parts(output.handle.clone(),expected.len()),expected.len()); }
    assert_eq!(read::<T>(output),expected);
}
#[test]
fn numeric_conversions_and_signed_remainders() {
    let data=[f64::NAN,f64::NEG_INFINITY,-1e30,-257.5,-128.9,-1.9,-0.,0.,1.9,127.9,255.9,65536.,1e30,f64::INFINITY];
    let input=tensor(&data);
    conversions(&input,&data.map(|x|x as i8)); conversions(&input,&data.map(|x|x as u8));
    conversions(&input,&data.map(|x|x as i16)); conversions(&input,&data.map(|x|x as u16));
    conversions(&input,&data.map(|x|x as i32)); conversions(&input,&data.map(|x|x as u32));
    conversions(&input,&data.map(|x|x as i64)); conversions(&input,&data.map(|x|x as u64));
    let input=tensor(&[-5,-4,-3,-2,-1,0,1,2,3,4,5]);
    let selected=device::select::select_if::<R,i32,Even>(&input,EvenLaunch::new(),32,false).unwrap();
    assert_eq!(read::<u64>(selected.count),[5]);
    assert_eq!(&read::<i32>(selected.values)[..5],[-4,-2,0,2,4]);
}

#[ruda(launch_unchecked, explicit_define)]
fn conversion_f32_kernel<T: Numeric>(input:&Array<f32>,output:&mut Array<T>,count:usize) {
    if ABSOLUTE_POS<count { output[ABSOLUTE_POS]=T::cast_from(input[ABSOLUTE_POS]); }
}

fn conversions_f32<T: TensorElement>(input:&RudaTensor<R>,expected:&[T]) {
    let output=ruda_kernel::tensor::allocation::empty_device_dtype(input.client.clone(),input.device.clone(),
        ruda_core::tensor::Shape::new([expected.len()]),<T as ruda_core::tensor::element::Element>::dtype());
    unsafe { conversion_f32_kernel::launch_unchecked::<T,R>(&input.client,RudaCount::Static(expected.len().div_ceil(32) as u32,1,1),RudaDim::new_1d(32),
        ArrayArg::from_raw_parts(input.handle.clone(),expected.len()),ArrayArg::from_raw_parts(output.handle.clone(),expected.len()),expected.len()); }
    assert_eq!(read::<T>(output),expected);
}

#[test]
fn numeric_conversion_boundaries_f32_f64() {
    let mut values=vec![f64::NAN,f64::INFINITY,f64::NEG_INFINITY,-0.0,0.0,-1.9,1.9];
    for bits in [7,8,15,16,31,32,63,64] {
        let upper=2.0f64.powi(bits);
        values.extend([upper.next_down(),upper,upper.next_up(),(-upper).next_down(),-upper,(-upper).next_up()]);
    }
    for chunk in values.chunks(32) {
        let input=tensor(chunk);
        conversions(&input,&chunk.iter().map(|&x|x as i8).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as u8).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as i16).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as u16).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as i32).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as u32).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as i64).collect::<Vec<_>>());
        conversions(&input,&chunk.iter().map(|&x|x as u64).collect::<Vec<_>>());
    }
    let mut values=vec![f32::NAN,f32::INFINITY,f32::NEG_INFINITY,-0.0,0.0,-1.9,1.9];
    for bits in [7,8,15,16,31,32,63,64] {
        let upper=2.0f32.powi(bits);
        values.extend([upper.next_down(),upper,upper.next_up(),(-upper).next_down(),-upper,(-upper).next_up()]);
    }
    let input=tensor(&values);
    conversions_f32(&input,&values.iter().map(|&x|x as i8).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as u8).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as i16).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as u16).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as i32).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as u32).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as i64).collect::<Vec<_>>());
    conversions_f32(&input,&values.iter().map(|&x|x as u64).collect::<Vec<_>>());
}
