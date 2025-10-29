use mistralrs_core::*;

use crate::{best_device, Model, TextModelBuilder, VisionModelBuilder};

/// Wrapper of model builders for LoRA models.
/// Supports both text and vision models.
pub struct LoraModelBuilder {
    inner: LoraModelBuilderInner,
    lora_adapter_ids: Vec<String>,
}

enum LoraModelBuilderInner {
    Text(TextModelBuilder),
    Vision(VisionModelBuilder),
}

impl LoraModelBuilder {
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        lora_adapter_ids: impl IntoIterator<Item = impl ToString>,
    ) -> Self {
        Self {
            inner: LoraModelBuilderInner::Text(text_model),
            lora_adapter_ids: lora_adapter_ids
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        }
    }

    pub fn from_vision_model_builder(
        vision_model: VisionModelBuilder,
        lora_adapter_ids: impl IntoIterator<Item = impl ToString>,
    ) -> Self {
        Self {
            inner: LoraModelBuilderInner::Vision(vision_model),
            lora_adapter_ids: lora_adapter_ids
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        }
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        match self.inner {
            LoraModelBuilderInner::Text(text_model) => {
                let config = NormalSpecificConfig {
                    topology: text_model.topology,
                    organization: text_model.organization,
                    write_uqff: text_model.write_uqff,
                    from_uqff: text_model.from_uqff,
                    imatrix: None,
                    calibration_file: None,
                    hf_cache_path: text_model.hf_cache_path,
                    matformer_config_path: None,
                    matformer_slice_name: None,
                };

                if text_model.with_logging {
                    initialize_logging();
                }

                let loader = NormalLoaderBuilder::new(
                    config,
                    text_model.chat_template,
                    text_model.tokenizer_json,
                    Some(text_model.model_id),
                    text_model.no_kv_cache,
                    text_model.jinja_explicit,
                )
                .with_lora(self.lora_adapter_ids)
                .build(text_model.loader_type)?;

                // Load, into a Pipeline
                let pipeline = loader.load_model_from_hf(
                    text_model.hf_revision,
                    text_model.token_source,
                    &text_model.dtype,
                    &text_model
                        .device
                        .unwrap_or(best_device(text_model.force_cpu)?),
                    !text_model.with_logging,
                    text_model
                        .device_mapping
                        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
                    text_model.isq,
                    text_model.paged_attn_cfg,
                )?;

                let scheduler_method = match text_model.paged_attn_cfg {
                    Some(_) => {
                        let config = pipeline
                            .lock()
                            .await
                            .get_metadata()
                            .cache_config
                            .as_ref()
                            .unwrap()
                            .clone();

                        SchedulerConfig::PagedAttentionMeta {
                            max_num_seqs: text_model.max_num_seqs,
                            config,
                        }
                    }
                    None => SchedulerConfig::DefaultScheduler {
                        method: DefaultSchedulerMethod::Fixed(text_model.max_num_seqs.try_into()?),
                    },
                };

                let mut runner = MistralRsBuilder::new(
                    pipeline,
                    scheduler_method,
                    text_model.throughput_logging,
                    text_model.search_bert_model,
                );
                if let Some(cb) = text_model.search_callback.clone() {
                    runner = runner.with_search_callback(cb);
                }
                for (name, cb) in &text_model.tool_callbacks {
                    runner = runner.with_tool_callback(name.clone(), cb.clone());
                }
                runner = runner
                    .with_no_kv_cache(text_model.no_kv_cache)
                    .with_no_prefix_cache(text_model.prefix_cache_n.is_none());

                if let Some(n) = text_model.prefix_cache_n {
                    runner = runner.with_prefix_cache_n(n)
                }

                Ok(Model::new(runner.build().await))
            }
            LoraModelBuilderInner::Vision(vision_model) => {
                let config = VisionSpecificConfig {
                    topology: vision_model.topology,
                    write_uqff: vision_model.write_uqff,
                    from_uqff: vision_model.from_uqff,
                    max_edge: vision_model.max_edge,
                    calibration_file: vision_model.calibration_file,
                    imatrix: vision_model.imatrix,
                    hf_cache_path: vision_model.hf_cache_path,
                    matformer_config_path: vision_model.matformer_config_path,
                    matformer_slice_name: vision_model.matformer_slice_name,
                };

                if vision_model.with_logging {
                    initialize_logging();
                }

                let loader = VisionLoaderBuilder::new(
                    config,
                    vision_model.chat_template,
                    vision_model.tokenizer_json,
                    Some(vision_model.model_id),
                    vision_model.jinja_explicit,
                )
                .with_lora(self.lora_adapter_ids)
                .build(vision_model.loader_type);

                // Load, into a Pipeline
                let device = vision_model
                    .device
                    .unwrap_or(best_device(vision_model.force_cpu)?);
                let pipeline = loader.load_model_from_hf(
                    vision_model.hf_revision,
                    vision_model.token_source,
                    &vision_model.dtype,
                    &device,
                    !vision_model.with_logging,
                    vision_model
                        .device_mapping
                        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_vision())),
                    vision_model.isq,
                    vision_model.paged_attn_cfg,
                )?;

                let scheduler_method = match vision_model.paged_attn_cfg {
                    Some(_) => {
                        let config = pipeline
                            .lock()
                            .await
                            .get_metadata()
                            .cache_config
                            .as_ref()
                            .cloned();

                        if let Some(config) = config {
                            SchedulerConfig::PagedAttentionMeta {
                                max_num_seqs: vision_model.max_num_seqs,
                                config,
                            }
                        } else {
                            SchedulerConfig::DefaultScheduler {
                                method: DefaultSchedulerMethod::Fixed(
                                    vision_model.max_num_seqs.try_into()?,
                                ),
                            }
                        }
                    }
                    None => SchedulerConfig::DefaultScheduler {
                        method: DefaultSchedulerMethod::Fixed(
                            vision_model.max_num_seqs.try_into()?,
                        ),
                    },
                };

                let mut runner = MistralRsBuilder::new(
                    pipeline,
                    scheduler_method,
                    vision_model.throughput_logging,
                    vision_model.search_bert_model,
                );
                if let Some(cb) = vision_model.search_callback.clone() {
                    runner = runner.with_search_callback(cb);
                }
                for (name, cb) in &vision_model.tool_callbacks {
                    runner = runner.with_tool_callback(name.clone(), cb.clone());
                }
                for (name, callback_with_tool) in &vision_model.tool_callbacks_with_tools {
                    runner = runner.with_tool_callback_and_tool(
                        name.clone(),
                        callback_with_tool.callback.clone(),
                        callback_with_tool.tool.clone(),
                    );
                }
                runner = runner
                    .with_no_kv_cache(false)
                    .with_no_prefix_cache(vision_model.prefix_cache_n.is_none());

                if let Some(n) = vision_model.prefix_cache_n {
                    runner = runner.with_prefix_cache_n(n)
                }

                Ok(Model::new(runner.build().await))
            }
        }
    }
}
