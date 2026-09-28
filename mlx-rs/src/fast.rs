//! Fast implementations of commonly used multi-op functions.

use std::ffi::CStr;

use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use mlx_internal_macros::generate_macro;

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
pub fn rope<'a>(
    array: impl AsRef<Array>,
    dimensions: i32,
    traditional: bool,
    base: impl Into<Option<f32>>,
    scale: f32,
    offset: i32,
    freqs: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rope`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rope`"
)]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        rope(array, dimensions, traditional, base, scale, offset, freqs)
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
pub fn rope_dynamic<'a>(
    array: impl AsRef<Array>,
    dimensions: i32,
    traditional: bool,
    base: impl Into<Option<f32>>,
    scale: f32,
    offset: impl AsRef<Array>,
    freqs: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rope_dynamic`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rope_dynamic`"
)]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        rope_dynamic(array, dimensions, traditional, base, scale, offset, freqs)
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    sinks: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            false,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scaled_dot_product_attention`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scaled_dot_product_attention`"
)]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scaled_dot_product_attention(queries, keys, values, scale, mask, sinks)
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: An optional multiplicative weight. When present, it must be one-dimensional with the
///   same size as the last axis of `x`. When absent, only normalization is applied.
/// - eps: A small additive constant for numerical stability
pub fn rms_norm(x: impl AsRef<Array>, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let empty_weight = weight
        .is_none()
        .then(|| unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) });
    let weight = weight
        .map(Array::as_ptr)
        .unwrap_or_else(|| empty_weight.as_ref().unwrap().as_ptr());
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight,
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rms_norm`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rms_norm`"
)]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || rms_norm(x, Some(weight.as_ref()), eps))
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
pub fn layer_norm<'a>(
    x: impl AsRef<Array>,
    weight: impl Into<Option<&'a Array>>,
    bias: impl Into<Option<&'a Array>>,
    eps: f32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`layer_norm`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `layer_norm`"
)]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || layer_norm(x, weight, bias, eps))
}

/// A template argument of a [`MetalKernel`]: a value substituted for a template parameter of its source.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateArg {
    /// A type parameter, substituted with the Metal name of the dtype.
    Dtype(crate::Dtype),
    /// An integer parameter.
    Int(i32),
    /// A boolean parameter.
    Bool(bool),
}

/// How one [`MetalKernel::apply`] call launches: its outputs, grid, threadgroup and template arguments.
///
/// This is the per-call half of `mx.fast.metal_kernel`: the Python binding takes these as keyword arguments of the
/// call, and a kernel built once is applied with a new configuration whenever its shapes change.
#[derive(Debug, Clone, PartialEq)]
pub struct MetalKernelConfig {
    outputs: Vec<(Vec<i32>, crate::Dtype)>,
    grid: (i32, i32, i32),
    thread_group: (i32, i32, i32),
    template_args: Vec<(String, TemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
}

impl MetalKernelConfig {
    /// A launch of `grid` threads (not threadgroups, as in Metal's `dispatchThreads`) in groups of
    /// `thread_group`, with no outputs yet.
    pub fn new(grid: (i32, i32, i32), thread_group: (i32, i32, i32)) -> Self {
        Self {
            outputs: Vec::new(),
            grid,
            thread_group,
            template_args: Vec::new(),
            init_value: None,
            verbose: false,
        }
    }

    /// Adds an output of the given shape and dtype, in the order of the kernel's output names.
    pub fn output(mut self, shape: &[i32], dtype: crate::Dtype) -> Self {
        self.outputs.push((shape.to_vec(), dtype));
        self
    }

    /// Adds a template argument.
    pub fn template_arg(mut self, name: impl Into<String>, value: TemplateArg) -> Self {
        self.template_args.push((name.into(), value));
        self
    }

    /// Fills every output with `value` before the kernel runs; without it the outputs start uninitialized.
    pub fn init_value(mut self, value: f32) -> Self {
        self.init_value = Some(value);
        self
    }

    /// Prints the generated Metal source when the kernel is built.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Frees an `mlx_fast_metal_kernel_config` when it goes out of scope.
struct RawMetalKernelConfig(mlx_sys::mlx_fast_metal_kernel_config);

impl Drop for RawMetalKernelConfig {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

/// Turns a status from mlx-c into a `Result`, reading the error it recorded.
fn check_status(status: i32, operation: &str) -> Result<()> {
    crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
    if status == crate::utils::SUCCESS {
        Ok(())
    } else {
        Err(crate::error::exception_from_status(status, operation))
    }
}

/// A NUL-terminated copy of `value`, refusing an interior NUL.
fn c_string(value: &str, what: &str) -> Result<std::ffi::CString> {
    std::ffi::CString::new(value)
        .map_err(|_| crate::error::Exception::custom(format!("{what} contains a NUL byte")))
}

impl MetalKernelConfig {
    fn to_raw(&self) -> Result<RawMetalKernelConfig> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let raw = RawMetalKernelConfig(unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() });
        if raw.0.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                crate::utils::SUCCESS + 1,
                "creating a Metal kernel configuration",
            ));
        }
        for (shape, dtype) in &self.outputs {
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    raw.0,
                    shape.as_ptr(),
                    shape.len(),
                    *dtype as mlx_sys::mlx_dtype,
                )
            };
            check_status(status, "adding a Metal kernel output")?;
        }
        let (x, y, z) = self.grid;
        check_status(
            unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_grid(raw.0, x, y, z) },
            "setting a Metal kernel's grid",
        )?;
        let (x, y, z) = self.thread_group;
        check_status(
            unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(raw.0, x, y, z) },
            "setting a Metal kernel's threadgroup",
        )?;
        for (name, value) in &self.template_args {
            let name = c_string(name, "a template argument's name")?;
            let status = match value {
                TemplateArg::Dtype(dtype) => unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        raw.0,
                        name.as_ptr(),
                        *dtype as mlx_sys::mlx_dtype,
                    )
                },
                TemplateArg::Int(value) => unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                        raw.0,
                        name.as_ptr(),
                        *value,
                    )
                },
                TemplateArg::Bool(value) => unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                        raw.0,
                        name.as_ptr(),
                        *value,
                    )
                },
            };
            check_status(status, "adding a Metal kernel template argument")?;
        }
        if let Some(value) = self.init_value {
            check_status(
                unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_init_value(raw.0, value) },
                "setting a Metal kernel's initial value",
            )?;
        }
        check_status(
            unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_verbose(raw.0, self.verbose) },
            "setting a Metal kernel's verbosity",
        )?;
        Ok(raw)
    }
}

/// Frees an `mlx_vector_string` when it goes out of scope.
struct RawVectorString(mlx_sys::mlx_vector_string);

impl RawVectorString {
    fn new(values: &[&str], what: &str) -> Result<Self> {
        let owned = values
            .iter()
            .map(|value| c_string(value, what))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<*const std::ffi::c_char> =
            owned.iter().map(|value| value.as_ptr()).collect();
        // mlx-c copies the strings into its own vector, so `owned` may go when this returns.
        let raw =
            unsafe { mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                crate::utils::SUCCESS + 1,
                "creating an MLX string vector",
            ));
        }
        Ok(Self(raw))
    }
}

impl Drop for RawVectorString {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_vector_string_free(self.0) };
    }
}

/// A custom Metal kernel: `mx.fast.metal_kernel`.
///
/// `source` is the body of the kernel; MLX generates its signature from the input and output names and from what
/// the body uses (`thread_position_in_grid`, `a_shape`, …), and compiles it the first time it is applied with a
/// given configuration. See MLX's documentation of `mx.fast.metal_kernel` for the conventions.
///
/// A kernel holds only the strings it was built from, so it can be shared between threads; each
/// [`apply`](Self::apply) builds a new lazy array on the calling thread's default stream.
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

// SAFETY: the kernel is an immutable function object over copies of its name, argument names and source; mlx-c
// never mutates it after construction, and applying it only reads it.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

impl MetalKernel {
    /// Builds a kernel named `name` over the given input and output names, with `source` as its body and `header`
    /// (possibly empty) placed before it.
    ///
    /// `ensure_row_contiguous` copies non-contiguous inputs before the launch; `atomic_outputs` declares the
    /// outputs `device atomic<T>*`.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let name = c_string(name, "a Metal kernel's name")?;
        let inputs = RawVectorString::new(input_names, "a Metal kernel input's name")?;
        let outputs = RawVectorString::new(output_names, "a Metal kernel output's name")?;
        let source = c_string(source, "a Metal kernel's source")?;
        let header = c_string(header, "a Metal kernel's header")?;
        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                inputs.0,
                outputs.0,
                source.as_ptr(),
                header.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                crate::utils::SUCCESS + 1,
                "building a Metal kernel",
            ));
        }
        Ok(Self { raw })
    }

    /// Applies the kernel to `inputs`, in the order of its input names, and returns its outputs in the order of
    /// its output names.
    pub fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        config: &MetalKernelConfig,
    ) -> Result<Vec<Array>> {
        let stream = Stream::thread_local_or_default();
        let inputs = crate::utils::VectorArray::try_from_iter(inputs.iter())?;
        let config = config.to_raw()?;
        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.raw,
                inputs.as_ptr(),
                config.0,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item_exact::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, Some(&weight), 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );

        let ones = Array::ones::<f32>(&[2, 4]).unwrap();
        let normalized = rms_norm(&ones, None, 0.0).unwrap();
        assert_eq!(normalized.as_slice::<f32>(), &[1.0; 8]);
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
                result.eval().unwrap();
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    #[test]
    fn test_metal_kernel_elementwise() {
        let kernel = MetalKernel::new(
            "tf_scaled_add",
            &["a", "b"],
            &["out"],
            r#"
                uint elem = thread_position_in_grid.x;
                out[elem] = a[elem] * T(SCALE) + b[elem];
            "#,
            "",
            true,
            false,
        )
        .unwrap();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let b = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4]);
        let config = MetalKernelConfig::new((4, 1, 1), (4, 1, 1))
            .output(&[4], crate::Dtype::Float32)
            .template_arg("T", TemplateArg::Dtype(crate::Dtype::Float32))
            .template_arg("SCALE", TemplateArg::Int(3));
        let out = kernel.apply(&[&a, &b], &config).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_slice::<f32>(), &[13.0, 26.0, 39.0, 52.0]);
    }

    #[test]
    fn test_metal_kernel_init_value_and_threads() {
        // Only the first two threads write; `init_value` fills the rest, and the kernel moves to another thread.
        let kernel = MetalKernel::new(
            "tf_partial_write",
            &["a"],
            &["out"],
            r#"
                uint elem = thread_position_in_grid.x;
                if (elem < 2) { out[elem] = a[elem]; }
            "#,
            "",
            true,
            false,
        )
        .unwrap();
        let out = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let a = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[4]);
                    let config = MetalKernelConfig::new((4, 1, 1), (4, 1, 1))
                        .output(&[4], crate::Dtype::Float32)
                        .init_value(-1.0);
                    kernel.apply(&[&a], &config).unwrap()[0]
                        .as_slice::<f32>()
                        .to_vec()
                })
                .join()
                .unwrap()
        });
        assert_eq!(out, vec![5.0, 6.0, -1.0, -1.0]);
    }

    #[test]
    fn test_metal_kernel_errors() {
        assert!(MetalKernel::new("bad\0name", &["a"], &["out"], "", "", true, false).is_err());
        let kernel = MetalKernel::new(
            "tf_does_not_compile",
            &["a"],
            &["out"],
            "this is not metal;",
            "",
            true,
            false,
        )
        .unwrap();
        let a = Array::from_slice(&[1.0f32], &[1]);
        let config =
            MetalKernelConfig::new((1, 1, 1), (1, 1, 1)).output(&[1], crate::Dtype::Float32);
        let result = kernel.apply(&[&a], &config).and_then(|out| out[0].eval());
        assert!(result.is_err());
    }
}
