use candle::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct LayerNormNoBias {
    weight: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl LayerNormNoBias {
    #[cfg(feature = "cuda")]
    pub fn forward_with_residual(
        &self,
        hidden_states: &Tensor,
        residual: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if matches!(hidden_states.device(), Device::Cuda(_)) {
            let shape = hidden_states.shape();
            let (normalized, residual) = candle_layer_norm::fused_add_layer_norm(
                &hidden_states.flatten_to(D::Minus2)?,
                &residual.flatten_to(D::Minus2)?,
                &self.weight,
                None,
                self.epsilon,
            )?;
            Ok((normalized.reshape(shape)?, residual.reshape(shape)?))
        } else {
            let residual = hidden_states.add(residual)?;
            Ok((self.forward(&residual, None)?, residual))
        }
    }

    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(hidden_size, "weight")
                .or_else(|_| vb.get(hidden_size, "gamma"))?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm-no-bias"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let mut hidden_states = hidden_states.clone();
                if let Some(residual) = residual {
                    hidden_states = hidden_states.add(residual)?;
                }
                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };
                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let mean_hidden_states =
                    (hidden_states.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states = hidden_states.broadcast_sub(&mean_hidden_states)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                let hidden_states = hidden_states_normed
                    .to_dtype(hidden_states_dtype)?
                    .broadcast_mul(&self.weight)?;

                Ok(hidden_states)
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_layer_norm::{fused_add_layer_norm, layer_norm};

                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;

                    let result = if let Some(residual) = residual {
                        let residual = residual.flatten_to(D::Minus2)?;

                        let (result, _) = fused_add_layer_norm(
                            &hidden_states,
                            &residual,
                            &self.weight,
                            None,
                            self.epsilon,
                        )?;
                        Ok(result)
                    } else {
                        layer_norm(&hidden_states, &self.weight, None, self.epsilon)
                    }?;
                    result.reshape(original_shape)
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(hidden_size, "weight")
                .or_else(|_| vb.get(hidden_size, "gamma"))?,
            bias: vb
                .get(hidden_size, "bias")
                .or_else(|_| vb.get(hidden_size, "beta"))?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let mut hidden_states = hidden_states.clone();
                if let Some(residual) = residual {
                    hidden_states = hidden_states.add(residual)?;
                }
                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };
                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let mean_hidden_states =
                    (hidden_states.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states = hidden_states.broadcast_sub(&mean_hidden_states)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                let hidden_states = hidden_states_normed
                    .to_dtype(hidden_states_dtype)?
                    .broadcast_mul(&self.weight)?;

                hidden_states.broadcast_add(&self.bias)
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_layer_norm::{fused_add_layer_norm, layer_norm};

                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;

                    let result = if let Some(residual) = residual {
                        let residual = residual.flatten_to(D::Minus2)?;

                        let (result, _) = fused_add_layer_norm(
                            &hidden_states,
                            &residual,
                            &self.weight,
                            Some(&self.bias),
                            self.epsilon,
                        )?;
                        Ok(result)
                    } else {
                        layer_norm(&hidden_states, &self.weight, Some(&self.bias), self.epsilon)
                    }?;
                    result.reshape(original_shape)
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn fused_layer_norm_preserves_residual_shape_and_values() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(123)?;
        let layer = LayerNormNoBias {
            weight: Tensor::ones(512, DType::F16, &device)?,
            epsilon: 1e-5,
            span: tracing::Span::none(),
        };
        let input = Tensor::randn(0f32, 1., (2, 3, 512), &device)?.to_dtype(DType::F16)?;
        let residual = Tensor::randn(0f32, 1., (2, 3, 512), &device)?.to_dtype(DType::F16)?;
        let sum = input.add(&residual)?;
        let expected = layer.forward(&sum, None)?;
        let (actual, carried) = layer.forward_with_residual(&input, &residual)?;
        assert_eq!(actual.dims(), input.dims());
        assert_eq!(carried.dims(), input.dims());
        let max_diff = |a: &Tensor, b: &Tensor| -> Result<f32> {
            (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()
        };
        assert_eq!(max_diff(&carried, &sum)?, 0.);
        // Fused normalization uses the FP32 sum before it is rounded to FP16.
        assert!(max_diff(&actual, &expected)? < 0.005);
        Ok(())
    }
}
