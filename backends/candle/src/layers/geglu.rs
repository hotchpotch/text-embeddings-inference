use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
use candle::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

struct GeGlu;

impl CustomOp1 for GeGlu {
    fn name(&self) -> &'static str {
        "tei-fused-geglu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("fused GeGLU requires CUDA F16")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let mut dims = layout.dims().to_vec();
        let width = *dims
            .last()
            .ok_or_else(|| candle::Error::Msg("GeGLU requires rank >= 1".into()))?;
        if width == 0 || width % 2 != 0 || !layout.is_contiguous() {
            candle::bail!("GeGLU requires contiguous input with an even positive last dimension");
        }
        let width = width / 2;
        *dims.last_mut().unwrap() = width;
        let shape = Shape::from(dims);
        let count = shape.elem_count();
        let device = storage.device();
        let input = storage.as_cuda_slice::<half::f16>()?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + count * 2);
        // Each output is initialized by one kernel thread.
        let output = unsafe { device.alloc::<half::f16>(count) }.map_err(candle::Error::wrap)?;
        if count != 0 {
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
            if width == 2048 && layout.start_offset() % 2 == 0 {
                let function = device.get_or_load_func("tei_geglu_2048_half2", ptx)?;
                unsafe {
                    function.launch(
                        LaunchConfig {
                            grid_dim: (((count / 2) as u32).div_ceil(512), 1, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&input, &output, count / 2),
                    )
                }
                .map_err(candle::Error::wrap)?;
            } else {
                let function = device.get_or_load_func("tei_geglu_f16", ptx)?;
                unsafe {
                    function.launch(
                        LaunchConfig::for_num_elems(count as u32),
                        (&input, &output, count, width),
                    )
                }
                .map_err(candle::Error::wrap)?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(output, device.clone()), shape))
    }
}

pub fn fused_geglu(input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        candle::bail!("fused GeGLU requires F16");
    }
    input.apply_op1_no_bwd(&GeGlu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, D};

    #[test]
    fn geglu_matches_unfused_for_offset_and_rank_three() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(73)?;
        let input = Tensor::randn(0f32, 1., (4, 17, 4096), &device)?
            .to_dtype(DType::F16)?
            .narrow(0, 1, 3)?;
        let a = input.narrow(D::Minus1, 0, 2048)?;
        let gate = input.narrow(D::Minus1, 2048, 2048)?;
        let expected = (a.gelu()? * gate)?;
        let actual = fused_geglu(&input)?;
        assert_eq!(actual.dims(), &[3, 17, 2048]);
        let diff = (expected.to_dtype(DType::F32)? - actual.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff <= 0.01, "max error {diff}");
        Ok(())
    }
}
