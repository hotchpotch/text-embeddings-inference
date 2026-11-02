//! Shape-specialized F16 GeGLU for base and xsmall ModernBERT.
//! Embedded SM89 PTX; no Python/Triton or external kernel directory at inference time.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};
use serde::Deserialize;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Deserialize)]
struct Metadata {
    abi: u32,
    name: String,
    block_m: u32,
    block_n: u32,
    num_warps: u32,
    shared: u32,
    hidden: usize,
    intermediate: usize,
}

struct Kernel {
    ptx: String,
    module: String,
    metadata: Metadata,
}

impl Kernel {
    fn load(name: &str, ptx: &str, json: &str) -> Result<Arc<Self>> {
        let metadata: Metadata = serde_json::from_str(json).map_err(candle::Error::wrap)?;
        if metadata.abi != 1
            || metadata.name != "dual_gemm"
            || !((metadata.hidden == 256 && metadata.intermediate == 1024)
                || (metadata.hidden == 512 && metadata.intermediate == 2048))
            || ![16, 32, 64, 128].contains(&metadata.block_m)
            || ![32, 64, 128].contains(&metadata.block_n)
            || ![4, 8].contains(&metadata.num_warps)
            || metadata.shared > 48 * 1024
        {
            candle::bail!("unsupported dual GeGLU kernel metadata");
        }
        let mut hash = DefaultHasher::new();
        ptx.hash(&mut hash);
        let module = format!("tei_dual_geglu_{name}_{:x}", hash.finish());
        Ok(Arc::new(Self {
            ptx: ptx.to_owned(),
            module,
            metadata,
        }))
    }
}

pub struct DualGeGlu {
    small: Arc<Kernel>,
    medium: Arc<Kernel>,
    large: Arc<Kernel>,
}

impl DualGeGlu {
    pub fn for_model(device: &Device, hidden: usize, intermediate: usize) -> Result<Option<Self>> {
        let Device::Cuda(cuda) = device else {
            return Ok(None);
        };
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let major = cuda
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(candle::Error::wrap)?;
        let minor = cuda
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(candle::Error::wrap)?;
        // Embedded PTX targets SM89 (including NVIDIA L4); older GPUs use the generic path.
        if major * 10 + minor < 89 {
            return Ok(None);
        }
        macro_rules! bundle {
            ($shape:literal) => {{
                macro_rules! kernel {
                    ($name:literal) => {
                        Kernel::load(
                            $name,
                            include_str!(concat!(
                                "../../kernels/modernbert/",
                                $shape,
                                "/",
                                $name,
                                ".ptx"
                            )),
                            include_str!(concat!(
                                "../../kernels/modernbert/",
                                $shape,
                                "/",
                                $name,
                                ".json"
                            )),
                        )?
                    };
                }
                Self {
                    small: kernel!("small"),
                    medium: kernel!("medium"),
                    large: kernel!("large"),
                }
            }};
        }
        let result = match (hidden, intermediate) {
            (256, 1024) => bundle!("xsmall"),
            (512, 2048) => bundle!("base"),
            _ => return Ok(None),
        };
        result.validate_shape(hidden, intermediate)?;
        for k in [&result.small, &result.medium, &result.large] {
            if k.metadata.hidden != hidden || k.metadata.intermediate != intermediate {
                candle::bail!("inconsistent dual GeGLU bundle dimensions");
            }
        }
        Ok(Some(result))
    }

    pub fn validate_shape(&self, hidden: usize, intermediate: usize) -> Result<()> {
        if hidden != self.small.metadata.hidden || intermediate != self.small.metadata.intermediate
        {
            candle::bail!("dual GeGLU bundle dimensions do not match model");
        }
        Ok(())
    }

    pub fn forward(&self, input: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(input.device(), Device::Cuda(_))
            || input.dtype() != DType::F16
            || input.rank() != 2
        {
            return Ok(None);
        }
        let (rows, hidden) = input.dims2()?;
        if hidden != self.small.metadata.hidden || rows == 0 || rows > i32::MAX as usize {
            return Ok(None);
        }
        let kernel = if rows <= 32 {
            &self.small
        } else if rows <= 128 {
            &self.medium
        } else if rows >= 2048 {
            &self.large
        } else {
            return Ok(None);
        };
        input
            .apply_op2_no_bwd(weight, &DualOp(kernel.clone()))
            .map(Some)
    }
}

struct DualOp(Arc<Kernel>);
impl CustomOp2 for DualOp {
    fn name(&self) -> &'static str {
        "dual-gemm-geglu"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("dual GeGLU requires CUDA F16");
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, hidden) = il.shape().dims2()?;
        let intermediate = self.0.metadata.intermediate;
        if hidden != self.0.metadata.hidden
            || wl.dims() != [2 * intermediate, hidden]
            || !il.is_contiguous()
            || !wl.is_contiguous()
            || input.device().id() != weight.device().id()
        {
            candle::bail!("invalid dual GeGLU inputs");
        }
        let device = input.device();
        let cuda = device.cuda_device();
        let input = input.as_cuda_slice::<half::f16>()?;
        let weight = weight.as_cuda_slice::<half::f16>()?;
        let input = input.slice(il.start_offset()..il.start_offset() + rows * hidden);
        let weight = weight.slice(wl.start_offset()..wl.start_offset() + 2 * intermediate * hidden);
        // The exported Triton ABI specializes all base pointers to 16-byte alignment.
        if *input.device_ptr() % 16 != 0 || *weight.device_ptr() % 16 != 0 {
            candle::bail!("dual GeGLU requires 16-byte aligned input and weight pointers");
        }
        let output = unsafe { device.alloc::<half::f16>(rows * intermediate) }
            .map_err(candle::Error::wrap)?;
        if !cuda.has_func(&self.0.module, "dual_gemm") {
            static LOAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _guard = LOAD_LOCK
                .lock()
                .map_err(|_| candle::Error::Msg("PTX load lock poisoned".into()))?;
            if !cuda.has_func(&self.0.module, "dual_gemm") {
                cuda.load_ptx(self.0.ptx.as_str().into(), &self.0.module, &["dual_gemm"])
                    .map_err(candle::Error::wrap)?;
            }
        }
        let function = cuda
            .get_func(&self.0.module, "dual_gemm")
            .ok_or_else(|| candle::Error::Msg("missing dual GeGLU kernel".into()))?;
        let m = &self.0.metadata;
        let launch = LaunchConfig {
            grid_dim: (
                (rows as u32).div_ceil(m.block_m),
                (intermediate as u32).div_ceil(m.block_n),
                1,
            ),
            block_dim: (m.num_warps * 32, 1, 1),
            shared_mem_bytes: m.shared,
        };
        // ABI v1: X, W, Y, rows, global scratch, profiling scratch; both scratch sizes are zero.
        unsafe { function.launch(launch, (&input, &weight, &output, rows as i32, 0u64, 0u64)) }
            .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((rows, intermediate)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dual_geglu_matches_reference_with_unequal_and_fallback_shapes() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(292)?;
        for (hidden, intermediate) in [(256, 1024), (512, 2048)] {
            let kernel = DualGeGlu::for_model(&device, hidden, intermediate)?
                .ok_or_else(|| candle::Error::Msg("SM89 or newer GPU required".into()))?;
            let intermediate = kernel.small.metadata.intermediate;
            let hidden = kernel.small.metadata.hidden;
            kernel.validate_shape(hidden, intermediate)?;
            assert!(kernel.validate_shape(hidden, intermediate + 1).is_err());
            assert!(kernel.validate_shape(hidden + 1, intermediate).is_err());
            let weight = Tensor::randn(0f32, 0.02, (2 * intermediate + 1, hidden), &device)?
                .to_dtype(DType::F16)?
                .narrow(0, 1, 2 * intermediate)?;
            let mut launched = 0;
            let mut fallbacks = 0;
            let mut max_abs = 0f32;
            for rows in [
                1, 3, 17, 31, 32, 33, 65, 127, 128, 129, 512, 2047, 2048, 2049, 4097,
            ] {
                let input = Tensor::randn(0f32, 1., (rows + 1, hidden), &device)?
                    .to_dtype(DType::F16)?
                    .narrow(0, 1, rows)?;
                let actual = kernel.forward(&input, &weight)?;
                if (129..2048).contains(&rows) {
                    assert!(actual.is_none());
                    fallbacks += 1;
                    continue;
                }
                let actual = actual.expect("eligible shape");
                let projected = input.matmul(&weight.t()?)?;
                let expected = crate::layers::fused_geglu(&projected)?;
                let diff = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(diff < 0.01, "rows {rows}: max error {diff}");
                max_abs = max_abs.max(diff);
                launched += 1;
            }
            assert_eq!((launched, fallbacks), (12, 3));
            println!("DUAL_GEGLU_SHAPE_CHECK {{\"hidden\":{hidden},\"intermediate\":{intermediate},\"launched\":{launched},\"fallbacks\":{fallbacks},\"max_abs_reference\":{max_abs}}}");
        }
        Ok(())
    }
}
