use super::{GenerationError,SamplingConfig,SamplingWorkspace,TokenSampler,ChaCha12Rng};
use ruda_model::record::{PrecisionSettings,Record};
use ruda_tensor::api::backend::Backend;

/// Durable request-local sampling configuration and exact ChaCha12 stream position.
/// Scratch probability/order buffers are not semantic state and are not recorded.
#[derive(Clone,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub struct TokenSamplerState {
    version: u32,
    config: SamplingConfig,
    rng: Vec<u8>,
}

impl<B: Backend> Record<B> for TokenSamplerState {
    type Item<S: PrecisionSettings> = Self;
    fn into_item<S: PrecisionSettings>(self) -> Self {self}
    fn from_item<S: PrecisionSettings>(item: Self,_device: &B::Device) -> Self {item}
}

impl TokenSamplerState {
    /// Original sampling options; seed identifies initialization, not current position.
    pub fn config(&self) -> SamplingConfig {self.config}

    /// Restore the actual next draw, without seeding again or consuming a draw.
    /// Native RNG bytes are tied to this record version and pinned chacha20 format.
    pub fn restore(self) -> Result<TokenSampler,GenerationError> {
        if self.version != 1 {return Err(GenerationError("unsupported token sampler record version".into()));}
        self.config.validate()?;
        let bytes = self.rng.as_slice().try_into()
            .map_err(|_|GenerationError("token sampler RNG state has an invalid byte count".into()))?;
        let rng = ChaCha12Rng::deserialize_state(bytes);
        if rng.serialize_state().as_slice() != self.rng.as_slice() {
            return Err(GenerationError("token sampler RNG state is not canonical".into()));
        }
        Ok(TokenSampler {config:self.config,rng,workspace:SamplingWorkspace::default()})
    }
}

impl TokenSampler {
    /// Actual options retained throughout this request.
    pub fn config(&self) -> SamplingConfig {self.config}

    /// Capture exact current RNG state, including the position of buffered draws.
    /// Does not clear scratch buffers, advance sampling or change the source sampler.
    pub fn to_state(&self) -> TokenSamplerState {
        TokenSamplerState {version:1,config:self.config,rng:self.rng.serialize_state().to_vec()}
    }

    /// Resume native request-local sampling from a checked exact RNG record.
    pub fn from_state(state: TokenSamplerState) -> Result<Self,GenerationError> {state.restore()}
}
