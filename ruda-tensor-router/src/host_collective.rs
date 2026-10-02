//! Explicit host-staged collectives across backend slots, optionally across nodes.
//! This is not an implicit fallback for NativeDistributedChannel. Each call takes
//! one contribution per LOCAL device; network ranks represent nodes, not devices.
//! Outputs are new tensors; inputs are never mutated. No autograd is implied.
use std::{fmt, marker::PhantomData, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Duration};
use ruda_tensor::{DType, DeviceId, DeviceOps, Shape, TensorData,
    distributed::ReduceOperation};
use ruccl::rank::{RankTransport, Opcode, ElementType, ANY_RANK};
use crate::{RouterTensor, RunnerChannel, RunnerClient};

/// A checked host bridge operation failed. Discard the group after any error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCollectiveError(pub String);
impl fmt::Display for HostCollectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for HostCollectiveError {}
fn error(message: impl Into<String>) -> HostCollectiveError { HostCollectiveError(message.into()) }

struct EpochGuard { aborted: Arc<AtomicBool>, committed: bool }
impl Drop for EpochGuard {
    fn drop(&mut self) { if !self.committed { self.aborted.store(true, Ordering::Release); } }
}

/// Opt-in FP32 host accumulation with FP16/BF16/FP32 input and output storage.
///
/// Different backend slots may participate. The caller supplies every local
/// device's tensor to ONE call, rather than calling once per local device.
/// A separate RankTransport session may join these local groups across nodes.
/// Both the host copies and their bandwidth cost are explicit in this API.
/// Timers require a Tokio runtime. Timeouts cannot preempt a blocked GPU driver.
pub struct HostStagedRouterGroup<R: RunnerChannel> {
    devices: Vec<DeviceId>,
    timeout: Duration,
    serial: tokio::sync::Mutex<()>,
    aborted: Arc<AtomicBool>,
    transport: Option<Arc<dyn RankTransport>>,
    _channel: PhantomData<R>,
}
impl<R: RunnerChannel> HostStagedRouterGroup<R> {
    /// Create a local group. Device IDs include the router's backend-slot bits.
    pub fn new(mut devices: Vec<DeviceId>, timeout: Duration) -> Result<Self, HostCollectiveError> {
        devices.sort();
        if devices.is_empty() || devices.windows(2).any(|w| w[0] == w[1]) || timeout.is_zero() {
            return Err(error("participants must be nonempty/unique and timeout positive"));
        }
        Ok(Self { devices, timeout, serial: tokio::sync::Mutex::new(()),
            aborted: Arc::new(AtomicBool::new(false)), transport: None, _channel: PhantomData })
    }

    /// Join local cross-backend groups via an already connected rank transport.
    /// Supply a dedicated session, not one shared with unrelated collectives.
    /// Every rank must call the same operations in the same order. Local device
    /// counts may differ: Mean is weighted by the GLOBAL number of devices.
    pub fn with_transport(mut self, transport: Arc<dyn RankTransport>) -> Result<Self, HostCollectiveError> {
        if transport.world_size() == 0 || transport.rank() >= transport.world_size() {
            return Err(error("invalid rank/world size"));
        }
        transport.set_timeout(self.timeout).map_err(|e| error(format!("transport timeout: {e:?}")))?;
        self.transport = Some(transport);
        Ok(self)
    }

    /// Mark the epoch unusable. Peers detect lost work via transport deadlines.
    /// An in-flight blocking transport worker may unwind until its own deadline.
    pub fn abort(&self) { self.aborted.store(true, Ordering::Release); }
    /// Whether this group has failed, timed out, or had a pending call dropped.
    pub fn is_aborted(&self) -> bool { self.aborted.load(Ordering::Acquire) }

    /// Sum/mean inputs and create one independent result on each original device.
    /// Returned order matches input order, regardless of the canonical sum order.
    /// Validation errors, cancellation, and deadlines poison the epoch. Callers
    /// must create a fresh group/transport rather than retry an ambiguous sum.
    pub async fn all_reduce(&self, tensors: &[RouterTensor<R::Client>], op: ReduceOperation)
        -> Result<Vec<RouterTensor<R::Client>>, HostCollectiveError>
    {
        if self.is_aborted() { return Err(error("host collective group aborted; create a fresh group")); }
        let mut guard = EpochGuard { aborted: self.aborted.clone(), committed: false };
        let operation = async {
            let _serial = self.serial.lock().await;
            if self.is_aborted() { return Err(error("host collective group aborted")); }
            let first = tensors.first().ok_or_else(|| error("no local contributions"))?;
            let shape = first.shape.clone();
            let dtype = first.dtype;
            if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
                || shape.is_empty() || shape.len() > 8 || shape.contains(&0)
            { return Err(error("host collective requires nonempty rank 1..8 FP16/BF16/FP32")); }
            let numel = shape.iter().try_fold(1usize, |a,&b| a.checked_mul(b))
                .ok_or_else(|| error("tensor element count overflow"))?;
            let mut order: Vec<_> = tensors.iter().enumerate()
                .map(|(i,t)| (t.client.device().id(),i)).collect();
            order.sort_by_key(|(id,_)| *id);
            if order.iter().map(|(id,_)| *id).collect::<Vec<_>>() != self.devices {
                return Err(error("exactly one contribution per local group device is required"));
            }
            if tensors.iter().any(|t| t.shape != shape || t.dtype != dtype) {
                return Err(error("contribution shape/dtype mismatch"));
            }
            let mut values = vec![0f32; numel];
            for (_,index) in order {
                let t = &tensors[index];
                let data = t.client.read_tensor_async(t.clone().into_ir()).await
                    .map_err(|e| error(format!("device read: {e:?}")))?;
                if data.shape != shape || data.dtype != dtype || data.num_elements() != numel {
                    return Err(error("device returned inconsistent tensor data"));
                }
                for (out,value) in values.iter_mut().zip(data.iter::<f32>()) { *out += value; }
                if self.is_aborted() { return Err(error("host collective aborted during copy")); }
            }
            if let Some(transport) = self.transport.clone() {
                let shape = shape.clone(); let count = tensors.len(); let aborted = self.aborted.clone();
                values = tokio::task::spawn_blocking(move || {
                    let result = exchange(&*transport, &shape, dtype, count, op, values);
                    if result.is_err() || aborted.load(Ordering::Acquire) {
                        let _ = transport.abort("host-staged router epoch failed/cancelled");
                        return Err(error("host-staged network operation failed or cancelled: ".to_owned()
                            + &format!("{:?}", result.as_ref().err())));
                    }
                    result
                }).await.map_err(|e| error(format!("transport worker: {e}")))??;
            } else if op == ReduceOperation::Mean {
                for value in &mut values { *value /= tensors.len() as f32; }
            }
            if self.is_aborted() { return Err(error("host collective aborted before publication")); }
            let data = TensorData::new(values, shape).convert_dtype(dtype);
            // The original storage remains unchanged even if uploading an output fails.
            let mut outputs = Vec::with_capacity(tensors.len());
            for t in tensors {
                outputs.push(t.client.register_tensor_data(data.clone()));
                t.client.sync().map_err(|e| error(format!("device upload: {e:?}")))?;
            }
            if self.is_aborted() { return Err(error("host collective aborted during publication")); }
            Ok(outputs)
        };
        let result = tokio::time::timeout(self.timeout, operation).await
            .map_err(|_| error("host collective deadline exceeded; no in-flight retry"))?;
        guard.committed = result.is_ok();
        result
    }
}

// Fixed-size little-endian metadata is gathered before the variable payload, so
// dtype/shape/operation mismatches cannot silently turn into a numerical result.
const HEADER_WORDS: usize = 13;
fn metadata(shape: &Shape, dtype: DType, count: usize, op: ReduceOperation) -> Vec<u8> {
    let mut words = [0u64; HEADER_WORDS];
    words[0] = 1; words[1] = if op == ReduceOperation::Sum { 0 } else { 1 };
    words[2] = match dtype { DType::F32 => 0, DType::F16 => 1, DType::BF16 => 2, _ => unreachable!() };
    words[3] = shape.len() as u64; words[4] = count as u64;
    for (slot,&dim) in words[5..].iter_mut().zip(shape.iter()) { *slot = dim as u64; }
    words.into_iter().flat_map(u64::to_le_bytes).collect()
}
fn validate_metadata(bytes: &[u8], own: &[u8], world: usize) -> Result<u64, HostCollectiveError> {
    let width = HEADER_WORDS * 8;
    if bytes.len() != world.checked_mul(width).ok_or_else(|| error("metadata length overflow"))? {
        return Err(error("rank metadata payload length mismatch"));
    }
    let mut total = 0u64;
    for peer in bytes.chunks_exact(width) {
        // Local device count is the only permitted difference.
        if peer[..32] != own[..32] || peer[40..] != own[40..] {
            return Err(error("rank operation/shape/dtype mismatch"));
        }
        let count = u64::from_le_bytes(peer[32..40].try_into().unwrap());
        if count == 0 { return Err(error("rank has no local devices")); }
        total = total.checked_add(count).ok_or_else(|| error("global device count overflow"))?;
    }
    Ok(total)
}
fn exchange(transport: &dyn RankTransport, shape: &Shape, dtype: DType, count: usize,
    op: ReduceOperation, local: Vec<f32>) -> Result<Vec<f32>, HostCollectiveError>
{
    let own = metadata(shape, dtype, count, op);
    let reply = transport.exchange(Opcode::AllGather, ElementType::U8, ANY_RANK,
        own.len() as u64, own.clone()).map_err(|e| error(format!("metadata exchange: {e:?}")))?;
    let world = transport.world_size() as usize;
    let total = validate_metadata(&reply.payload, &own, world)?;
    let bytes: Vec<_> = local.iter().flat_map(|v| v.to_le_bytes()).collect();
    let width = bytes.len();
    let reply = transport.exchange(Opcode::AllGather, ElementType::U8, ANY_RANK,
        width as u64, bytes).map_err(|e| error(format!("data exchange: {e:?}")))?;
    if reply.payload.len() != width.checked_mul(world).ok_or_else(|| error("payload length overflow"))? {
        return Err(error("gathered data length mismatch"));
    }
    let mut result = vec![0f32; local.len()];
    for peer in reply.payload.chunks_exact(width) {
        for (out, bytes) in result.iter_mut().zip(peer.chunks_exact(4)) {
            *out += f32::from_le_bytes(bytes.try_into().unwrap());
        }
    }
    if op == ReduceOperation::Mean { for value in &mut result { *value /= total as f32; } }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DirectByteChannel, duo::MultiDevice, get_client};
    type Host = ruda_tensor_host::Host;
    type Channel = DirectByteChannel<(Host, Host)>;
    fn participants() -> Vec<RouterTensor<<Channel as RunnerChannel>::Client>> {
        [MultiDevice::B1(Default::default()), MultiDevice::B2(Default::default())]
            .iter().enumerate().map(|(i,d)| get_client::<Channel>(d)
                .register_tensor_data(TensorData::new(vec![i as f32 + 1., 3.], [2]))).collect()
    }
    #[tokio::test]
    async fn different_slots_sum_mean_and_inputs_unchanged() {
        let values = participants();
        let group = HostStagedRouterGroup::<Channel>::new(values.iter().map(|t| t.client.device().id()).collect(),
            Duration::from_secs(10)).unwrap();
        for (op,expected) in [(ReduceOperation::Sum,vec![3.,6.]),(ReduceOperation::Mean,vec![1.5,3.])] {
            for result in group.all_reduce(&values,op).await.unwrap() {
                assert_eq!(result.into_data().await.unwrap().iter::<f32>().collect::<Vec<_>>(), expected);
            }
        }
        assert_eq!(values[0].clone().into_data().await.unwrap().iter::<f32>().collect::<Vec<_>>(),vec![1.,3.]);
        assert!(!group.is_aborted());
    }
    #[tokio::test]
    async fn wrong_membership_poisoned_no_retry() {
        let values = participants(); let ids = values.iter().map(|t| t.client.device().id()).collect();
        let group = HostStagedRouterGroup::<Channel>::new(ids,Duration::from_secs(10)).unwrap();
        assert!(group.all_reduce(&values[..1],ReduceOperation::Sum).await.is_err());
        assert!(group.is_aborted());
        assert!(group.all_reduce(&values,ReduceOperation::Sum).await.is_err());
    }
    #[test]
    fn rank_header_weights_unequal_device_counts_and_rejects_mismatch() {
        let a=metadata(&Shape::from([2,3]),DType::F32,2,ReduceOperation::Mean);
        let b=metadata(&Shape::from([2,3]),DType::F32,1,ReduceOperation::Mean);
        assert_eq!(validate_metadata(&[a.clone(),b].concat(),&a,2).unwrap(),3);
        let bad=metadata(&Shape::from([2,3]),DType::F32,1,ReduceOperation::Sum);
        assert!(validate_metadata(&[a.clone(),bad].concat(),&a,2).is_err());
        assert!(validate_metadata(&a,&a,2).is_err());
    }
    #[test]
    fn cancelled_epoch_stays_failed() {
        let aborted=Arc::new(AtomicBool::new(false));
        drop(EpochGuard {aborted:aborted.clone(),committed:false});
        assert!(aborted.load(Ordering::Acquire));
    }
    #[test]
    fn real_tcp_cross_node_mean_counts_devices_not_nodes() {
        use ruccl::rank::{TcpRendezvousServer, TcpRankSession, UniqueId};
        let id=UniqueId::from_bytes([0x91;16]);
        let server=TcpRendezvousServer::bind("127.0.0.1:0",id,2).unwrap()
            .with_collective_timeout(Duration::from_secs(5)).unwrap();
        let address=server.local_addr().unwrap();
        let coordinator=std::thread::spawn(move || server.run());
        let ranks=(0..2).map(|rank|std::thread::spawn(move || {
            let session=TcpRankSession::connect(address,id,rank,2,Duration::from_secs(5)).unwrap();
            // Node 0 sums two devices (1+2); node 1 has one device (9).
            let (count,local)=if rank==0 {(2,vec![3f32,6.])} else {(1,vec![9f32,18.])};
            let result=exchange(&session,&Shape::from([2]),DType::F32,count,ReduceOperation::Mean,local).unwrap();
            assert_eq!(result,vec![4.,8.]);
        })).collect::<Vec<_>>();
        for rank in ranks {rank.join().unwrap();}
        let _=coordinator.join().unwrap();
    }

}
