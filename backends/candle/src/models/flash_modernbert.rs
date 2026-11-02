use std::collections::HashMap;

use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, index_select, LayerNormNoBias, Linear};
use crate::models::modernbert::{
    ClassificationHead, ModernBertClassificationHead, ModernBertConfig, ModernBertEmbeddings,
    ModernBertMLP,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,
    local_attention: usize,
    use_local_attention: bool,

    span: tracing::Span,
}

impl ModernBertAttention {
    pub fn load(vb: VarBuilder, index: usize, config: &ModernBertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let wqkv_weight = vb
            .pp("Wqkv")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let wqkv_bias = if config.attention_bias {
            vb.pp("Wqkv").get(hidden_size * 3, "bias").ok()
        } else {
            None
        };
        let wqkv: Linear = Linear::new(wqkv_weight, wqkv_bias, None);

        let wo_weight = vb.pp("Wo").get((hidden_size, hidden_size), "weight")?;
        let wo_bias = if config.attention_bias {
            vb.pp("Wo").get(hidden_size, "bias").ok()
        } else {
            None
        };
        let wo = Linear::new(wo_weight, wo_bias, None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        let use_local_attention = index % config.global_attn_every_n_layers != 0;

        Ok(Self {
            wqkv,
            wo,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            local_attention: config.local_attention / 2 as usize,
            use_local_attention,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        cls_queries: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.wqkv.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;

        // Split qkv tensor
        let q = qkv.narrow(1, 0, self.num_attention_heads)?;
        let k = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let v = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;

        apply_rotary_inplace(&q, &k, &cos, &sin, true)?;

        // In the final global-attention layer, CLS pooling needs only one query per sequence.
        let q = if let Some((indices, _)) = cls_queries {
            index_select(&qkv, indices, 0)?.narrow(1, 0, self.num_attention_heads)?
        } else {
            q
        };
        let (cu_seqlens_q, max_s_q) = cls_queries
            .map(|(_, lengths)| (lengths, 1))
            .unwrap_or((cu_seqlens, max_s));

        let window_size = if self.use_local_attention {
            Some(self.local_attention)
        } else {
            None
        };

        let attention = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens_q,
            cu_seqlens,
            max_s_q,
            max_s,
            self.softmax_scale,
            false,
            window_size,
            window_size,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.wo.forward(&attention)?;

        Ok(hidden_states)
    }
}

struct ModernBertEncoderLayer {
    attn_norm: Option<LayerNormNoBias>,
    attn: ModernBertAttention,
    mlp_norm: LayerNormNoBias,
    mlp: ModernBertMLP,

    span: tracing::Span,
}

impl ModernBertEncoderLayer {
    pub fn load(vb: VarBuilder, index: usize, config: &ModernBertConfig) -> Result<Self> {
        let attn_norm = if index != 0 {
            Some(LayerNormNoBias::load(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )?)
        } else {
            None
        };

        let attn = ModernBertAttention::load(vb.pp("attn"), index, config)?;

        let mlp_norm = LayerNormNoBias::load(
            vb.pp("mlp_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;

        let span = tracing::span!(tracing::Level::TRACE, "layer");

        Ok(ModernBertEncoderLayer {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        cls_queries: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (attn_norm, residual) = if let Some(attn_norm) = &self.attn_norm {
            if let Some(residual) = residual {
                attn_norm.forward_with_residual(hidden_states, residual)?
            } else {
                (
                    attn_norm.forward(hidden_states, None)?,
                    hidden_states.clone(),
                )
            }
        } else {
            (hidden_states.clone(), hidden_states.clone())
        };

        let residual = if let Some((indices, _)) = cls_queries {
            index_select(&residual, indices, 0)?
        } else {
            residual
        };
        let attn_outputs =
            self.attn
                .forward(&attn_norm, cu_seqlens, cos, sin, max_s, cls_queries)?;

        let (normalized, residual) = self
            .mlp_norm
            .forward_with_residual(&attn_outputs, &residual)?;
        Ok((self.mlp.forward(&normalized)?, residual))
    }
}

struct ModernBertEncoder {
    layers: Vec<ModernBertEncoderLayer>,

    global_attn_every_n_layers: usize,

    span: tracing::Span,
}

impl ModernBertEncoder {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| ModernBertEncoderLayer::load(vb.pp(format!("{index}")), index, config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(ModernBertEncoder {
            layers,
            global_attn_every_n_layers: config.global_attn_every_n_layers,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        rotary_cache: &HashMap<bool, (Tensor, Tensor)>,
        max_s: usize,
        cls_queries: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();
        let mut residual = None;

        for (index, layer) in self.layers.iter().enumerate() {
            let use_local_attention = index % self.global_attn_every_n_layers != 0;
            let (cos, sin) = &rotary_cache[&use_local_attention];

            let queries = if index + 1 == self.layers.len() {
                cls_queries
            } else {
                None
            };
            let (output, updated_residual) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                cu_seqlens,
                cos,
                sin,
                max_s,
                queries,
            )?;
            hidden_states = output;
            residual = Some(updated_residual);
        }

        Ok((
            hidden_states,
            residual.expect("ModernBERT must have at least one layer"),
        ))
    }
}

pub struct FlashModernBertModel {
    embeddings: ModernBertEmbeddings,
    encoder: ModernBertEncoder,
    final_norm: LayerNormNoBias,
    pool: Pool,
    classifier: Option<Box<dyn ClassificationHead + Send>>,

    rotary_cache: HashMap<bool, (Tensor, Tensor)>,

    device: Device,

    span: tracing::Span,
}

impl FlashModernBertModel {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashModernBert requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashModernBert requires DType::F16")
        }

        let (pool, classifier) = match model_type {
            ModelType::Classifier => {
                let pool: Pool = config.classifier_pooling.clone().unwrap_or(Pool::Cls);

                let classifier: Box<dyn ClassificationHead + Send> =
                    Box::new(ModernBertClassificationHead::load(vb.clone(), config)?);

                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for ModernBert")
                }

                (pool, None)
            }
        };

        let embeddings = ModernBertEmbeddings::load(vb.pp("model.embeddings"), config)
            .or_else(|_| ModernBertEmbeddings::load(vb.pp("embeddings"), config))?;
        let encoder = ModernBertEncoder::load(vb.pp("model.layers"), config)
            .or_else(|_| ModernBertEncoder::load(vb.pp("layers"), config))?;
        let final_norm = LayerNormNoBias::load(
            vb.pp("model.final_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )
        .or_else(|_| {
            LayerNormNoBias::load(
                vb.pp("final_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )
        })?;

        let rotary_dim = config.hidden_size / config.num_attention_heads;
        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();

        for use_local_attention in [true, false] {
            let rope_theta = if use_local_attention {
                config.local_rope_theta
            } else {
                config.global_rope_theta
            };

            let max_position_embeddings = config.max_position_embeddings;

            let inv_freqs = get_inv_freqs(rotary_dim, rope_theta as f32, vb.device(), None)?;

            let (cos, sin) = get_cos_sin(max_position_embeddings, &inv_freqs, vb.dtype(), false)?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        Ok(Self {
            embeddings,
            encoder,
            final_norm,
            pool,
            classifier,
            rotary_cache,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let shape = batch.input_ids.len();
        let max_length = batch.max_length as usize;

        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();
        for use_local_attention in [true, false] {
            let (cos, sin) = &self.rotary_cache[&use_local_attention];

            let cos = index_select(&cos, &position_ids, 0)?;
            let sin = index_select(&sin, &position_ids, 0)?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        let hidden_states = self.embeddings.forward(&input_ids)?;
        let prune_cls = self.pool == Pool::Cls
            && batch.raw_indices.is_empty()
            && batch.pooled_indices.len() == batch_size
            && (self.encoder.layers.len() - 1) % self.encoder.global_attn_every_n_layers == 0;
        let cls_queries = if prune_cls {
            Some((
                cu_seqlens.narrow(0, 0, batch_size)?,
                Tensor::arange(0u32, batch_size as u32 + 1, &self.device)?,
            ))
        } else {
            None
        };
        let (hidden_states, residual) = self.encoder.forward(
            &hidden_states,
            &cu_seqlens,
            &rotary_cache,
            max_length,
            cls_queries
                .as_ref()
                .map(|(indices, lengths)| (indices, lengths)),
        )?;
        let outputs = self.final_norm.forward(&hidden_states, Some(&residual))?;

        if prune_cls {
            return Ok((Some(outputs), None));
        }

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        if has_raw_requests {
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            indices = index_select(&indices, &pooled_indices, 0)?
                        }

                        Some(index_select(&outputs, &indices, 0)?)
                    } else {
                        Some(
                            match self.pool {
                                Pool::Cls => outputs.i(0)?,
                                Pool::LastToken => {
                                    outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                                }
                                _ => unreachable!(),
                            }
                            .unsqueeze(0)?,
                        )
                    }
                }
                Pool::Mean => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .into_iter()
                            .map(|i| {
                                let i = i as usize;
                                let start = batch.cumulative_seq_lengths[i];
                                let len = batch.cumulative_seq_lengths[i + 1] - start;

                                // Mean
                                let embeddings = outputs.narrow(0, start as usize, len as usize)?;
                                embeddings.sum_keepdim(0)? / (len as f64)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some((outputs.sum_keepdim(0)? / (batch.max_length as f64))?)
                    }
                }
                Pool::Splade => {
                    unreachable!();
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                Some(index_select(&outputs, &final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for FlashModernBertModel {
    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}

#[cfg(all(test, feature = "flash-attn"))]
mod tests {
    use super::*;

    #[test]
    fn final_cls_queries_match_full_attention_for_unequal_lengths() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let weight =
            |rows, cols| Tensor::randn(0f32, 0.05, (rows, cols), &device)?.to_dtype(DType::F16);
        let attention = ModernBertAttention {
            wqkv: Linear::new(weight(192, 64)?, None, None),
            wo: Linear::new(weight(64, 64)?, None, None),
            num_attention_heads: 2,
            attention_head_size: 32,
            softmax_scale: 1. / 32f32.sqrt(),
            local_attention: 64,
            use_local_attention: false,
            span: tracing::Span::none(),
        };
        let input = Tensor::randn(0f32, 1., (11, 64), &device)?.to_dtype(DType::F16)?;
        let lengths = Tensor::new(&[0u32, 3, 11], &device)?;
        let indices = Tensor::new(&[0u32, 3], &device)?;
        let queries = Tensor::new(&[0u32, 1, 2], &device)?;
        let cos = Tensor::ones((11, 16), DType::F16, &device)?;
        let sin = Tensor::zeros((11, 16), DType::F16, &device)?;
        let full = attention.forward(&input, &lengths, &cos, &sin, 8, None)?;
        let expected = index_select(&full, &indices, 0)?;
        let actual =
            attention.forward(&input, &lengths, &cos, &sin, 8, Some((&indices, &queries)))?;
        assert_eq!(actual.dims(), &[2, 64]);
        let diff = (expected.to_dtype(DType::F32)? - actual.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 0.002, "CLS attention max difference: {diff}");
        Ok(())
    }
}
