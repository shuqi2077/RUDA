use core::num::NonZeroUsize;

/// Check that the parameter value is non-zero.
// NOTE: for now we keep usize but we could refactor the parameters to hold `NonZeroUsize`.
fn check_nonzero(value: usize, msg: &str) -> usize {
    NonZeroUsize::new(value).expect(msg);
    value
}

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvOptions<N> {
    /// Constructs a new `ConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }
}

/// Convolution options with support for asymmetric padding.
///
/// Wraps [`ConvOptions`] (which represents symmetric padding for the backend op)
/// and adds optional asymmetric padding. When asymmetric padding is specified,
/// the functional convolution layer applies an explicit pad operation before
/// dispatching to the backend.
///
/// Implements `From<ConvOptions<N>>` for backward compatibility.
#[derive(Debug, Clone)]
pub struct PaddedConvOptions<const N: usize> {
    /// The underlying convolution options for the backend.
    pub options: ConvOptions<N>,
    /// Padding at the end of each dimension (e.g., bottom/right for 2D).
    /// If `None`, padding is symmetric (same as `options.padding`).
    /// If `Some`, specifies different end-padding per dimension.
    pub padding_end: Option<[usize; N]>,
}

impl<const N: usize> PaddedConvOptions<N> {
    /// Creates options with asymmetric padding.
    ///
    /// `padding_start` is stored in `ConvOptions::padding`.
    /// `padding_end` specifies the end padding per dimension.
    pub fn asymmetric(
        stride: [usize; N],
        padding_start: [usize; N],
        padding_end: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        let options = ConvOptions::new(stride, padding_start, dilation, groups);
        if padding_start == padding_end {
            Self {
                options,
                padding_end: None,
            }
        } else {
            Self {
                options,
                padding_end: Some(padding_end),
            }
        }
    }

    /// Returns true if padding is asymmetric.
    pub fn is_asymmetric(&self) -> bool {
        self.padding_end.is_some()
    }
}

impl<const N: usize> From<ConvOptions<N>> for PaddedConvOptions<N> {
    fn from(options: ConvOptions<N>) -> Self {
        Self {
            options,
            padding_end: None,
        }
    }
}

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DeformConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Weight Groups (non-zero).
    pub weight_groups: usize,

    /// Offset Groups (non-zero).
    pub offset_groups: usize,

    /// End padding per dimension, or `None` to use symmetric `padding`.
    pub padding_end: Option<[usize; N]>,
}

impl<const N: usize> DeformConvOptions<N> {
    /// Constructs a new `DeformConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        weight_groups: usize,
        offset_groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            weight_groups: check_nonzero(weight_groups, "weight groups must be non-zero"),
            offset_groups: check_nonzero(offset_groups, "offset groups must be non-zero"),
            padding_end: None,
        }
    }

    /// Set end padding while retaining `padding` as the sampling grid's start padding.
    pub fn with_padding_end(mut self, padding_end: [usize; N]) -> Self {
        self.padding_end = (padding_end != self.padding).then_some(padding_end);
        self
    }

    /// Calculate output spatial dimensions with the configured dilation and padding.
    pub fn output_size(&self, input: [usize; N], kernel: [usize; N]) -> [usize; N] {
        let padding_end = self.padding_end.unwrap_or(self.padding);
        core::array::from_fn(|axis| {
            let extent = kernel[axis].checked_sub(1)
                .and_then(|size| size.checked_mul(self.dilation[axis]))
                .and_then(|size| size.checked_add(1))
                .expect("deform_conv kernel extent must be non-zero and fit usize");
            let padded = input[axis].checked_add(self.padding[axis])
                .and_then(|size| size.checked_add(padding_end[axis]))
                .expect("deform_conv padded input must fit usize");
            padded.checked_sub(extent)
                .expect("deform_conv kernel exceeds padded input") / self.stride[axis] + 1
        })
    }
}

/// Transposed convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvTransposeOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Padding out.
    pub padding_out: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvTransposeOptions<N> {
    /// Constructs a new `ConvTransposeOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        padding_out: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            padding_out,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }
}

/// Unfold operation options.
#[derive(Debug, Clone)]
pub struct UnfoldOptions {
    /// The number of positions to slide over the input tensor in each dimension.
    /// A stride of `[1, 1]` will slide the kernel one pixel at a time.
    pub stride: [usize; 2],

    /// The number of zero-padding pixels added to each side of the input tensor in each dimension.
    pub padding: [usize; 2],

    /// The spacing between the blocks (patches) in the original input tensor.
    pub dilation: [usize; 2],
}

impl UnfoldOptions {
    /// Constructs a new `UnfoldOptions`.
    pub fn new(stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2]) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
        }
    }
}
