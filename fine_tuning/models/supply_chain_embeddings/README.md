---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:272
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What is the risk level for ASE during COVID-19 Pandemic?
  sentences:
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    Germany. Company: Bosch. Supply disruption index: 9.83. Export control level:
    5.05. Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown
    risk: 10.00.'
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    Taiwan. Company: ASE. Supply disruption index: 9.83. Export control level: 5.05.
    Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown risk:
    10.00.'
  - 'Semiconductor disruption: Russia-Ukraine / US Export Controls (HIGH). Year: 2022.
    Country: China. Company: JCET. Supply disruption index: 6.42. Export control level:
    4.90. Chip price index: 219.18. Natural disaster risk: 3.19. Factory shutdown
    risk: 2.95.'
- source_sentence: What is the risk level for Arm during Global Chip Shortage + Texas
    Freeze?
  sentences:
  - 'Semiconductor disruption: Global Chip Shortage + Texas Freeze (CRITICAL). Year:
    2021. Country: UK. Company: Arm. Supply disruption index: 7.30. Export control
    level: 2.35. Chip price index: 199.25. Natural disaster risk: 5.47. Factory shutdown
    risk: 4.09.'
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    UK. Company: Graphcore. Supply disruption index: 9.83. Export control level: 5.05.
    Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown risk:
    10.00.'
  - 'Semiconductor disruption: Global Chip Shortage + Texas Freeze (CRITICAL). Year:
    2021. Country: China. Company: JCET. Supply disruption index: 7.30. Export control
    level: 2.35. Chip price index: 199.25. Natural disaster risk: 5.47. Factory shutdown
    risk: 4.09.'
- source_sentence: Electronics supply disruption Japan 2023 AI Demand Surge + Inventory
    Correction
  sentences:
  - 'Semiconductor disruption: AI Demand Surge + Inventory Correction (HIGH). Year:
    2023. Country: China. Company: SMIC. Supply disruption index: 5.84. Export control
    level: 3.59. Chip price index: 241.09. Natural disaster risk: 5.15. Factory shutdown
    risk: 7.93.'
  - 'Historical semiconductor disruption signal

    Year: 2023

    Country: USA

    Company: Applied Materials

    Event: AI Demand Surge + Inventory Correction

    Known severity: HIGH

    Supply disruption index: 5.837

    Semiconductor security risk: 5.464

    Natural disaster risk: 5.148

    Factory shutdown risk: 7.926

    Export control level: 3.591

    Chip price index: 241.092'
  - 'Semiconductor disruption: AI Demand Surge + Inventory Correction (HIGH). Year:
    2023. Country: Japan. Company: Tokyo Electron. Supply disruption index: 5.84.
    Export control level: 3.59. Chip price index: 241.09. Natural disaster risk: 5.15.
    Factory shutdown risk: 7.93.'
- source_sentence: What is the risk level for Ansys during AI Demand Surge + Inventory
    Correction?
  sentences:
  - 'Semiconductor disruption: AI Memory Shortage (HBM) (HIGH). Year: 2024. Country:
    USA. Company: Qualcomm. Supply disruption index: 7.14. Export control level: 1.47.
    Chip price index: 265.20. Natural disaster risk: 5.47. Factory shutdown risk:
    7.29.'
  - 'Semiconductor disruption: AI Demand Surge + Inventory Correction (HIGH). Year:
    2023. Country: USA. Company: Ansys. Supply disruption index: 5.84. Export control
    level: 3.59. Chip price index: 241.09. Natural disaster risk: 5.15. Factory shutdown
    risk: 7.93.'
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    Switzerland. Company: STMicroelectronics. Supply disruption index: 9.83. Export
    control level: 5.05. Chip price index: 181.14. Natural disaster risk: 9.82. Factory
    shutdown risk: 10.00.'
- source_sentence: Electronics supply disruption Japan 2020 COVID-19 Pandemic
  sentences:
  - 'Semiconductor disruption: Global Chip Shortage + Texas Freeze (CRITICAL). Year:
    2021. Country: Japan. Company: Kioxia. Supply disruption index: 7.30. Export control
    level: 2.35. Chip price index: 199.25. Natural disaster risk: 5.47. Factory shutdown
    risk: 4.09.'
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    Japan. Company: Tokyo Electron. Supply disruption index: 9.83. Export control
    level: 5.05. Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown
    risk: 10.00.'
  - 'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country:
    USA. Company: Skyworks. Supply disruption index: 9.83. Export control level: 5.05.
    Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown risk:
    10.00.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: supply chain ir
      type: supply_chain_ir
    metrics:
    - type: cosine_accuracy@1
      value: 0.7272727272727273
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.8051948051948052
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8311688311688312
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8701298701298701
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.7272727272727273
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.2683982683982683
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.16623376623376618
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08701298701298699
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.7272727272727273
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.8051948051948052
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8311688311688312
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8701298701298701
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7959033970056415
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.7725108225108225
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.7771409847588931
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Electronics supply disruption Japan 2020 COVID-19 Pandemic',
    'Semiconductor disruption: COVID-19 Pandemic (CRITICAL). Year: 2020. Country: Japan. Company: Tokyo Electron. Supply disruption index: 9.83. Export control level: 5.05. Chip price index: 181.14. Natural disaster risk: 9.82. Factory shutdown risk: 10.00.',
    'Semiconductor disruption: Global Chip Shortage + Texas Freeze (CRITICAL). Year: 2021. Country: Japan. Company: Kioxia. Supply disruption index: 7.30. Export control level: 2.35. Chip price index: 199.25. Natural disaster risk: 5.47. Factory shutdown risk: 4.09.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7789, 0.3727],
#         [0.7789, 1.0000, 0.4923],
#         [0.3727, 0.4923, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `supply_chain_ir`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.7273     |
| cosine_accuracy@3   | 0.8052     |
| cosine_accuracy@5   | 0.8312     |
| cosine_accuracy@10  | 0.8701     |
| cosine_precision@1  | 0.7273     |
| cosine_precision@3  | 0.2684     |
| cosine_precision@5  | 0.1662     |
| cosine_precision@10 | 0.087      |
| cosine_recall@1     | 0.7273     |
| cosine_recall@3     | 0.8052     |
| cosine_recall@5     | 0.8312     |
| cosine_recall@10    | 0.8701     |
| **cosine_ndcg@10**  | **0.7959** |
| cosine_mrr@10       | 0.7725     |
| cosine_map@100      | 0.7771     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 272 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 272 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 17.61 tokens</li><li>max: 24 tokens</li></ul> | <ul><li>min: 36 tokens</li><li>mean: 69.65 tokens</li><li>max: 100 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                      | sentence_1                                                                                                                                                                                                                                                                                   |
  |:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the risk level for Entegris during AI Demand Surge + Inventory Correction?</code> | <code>Semiconductor disruption: AI Demand Surge + Inventory Correction (HIGH). Year: 2023. Country: USA. Company: Entegris. Supply disruption index: 5.84. Export control level: 3.59. Chip price index: 241.09. Natural disaster risk: 5.15. Factory shutdown risk: 7.93.</code>            |
  | <code>What is the risk level for Cerebras during Russia-Ukraine / US Export Controls?</code>    | <code>Semiconductor disruption: Russia-Ukraine / US Export Controls (HIGH). Year: 2022. Country: USA. Company: Cerebras. Supply disruption index: 6.42. Export control level: 4.90. Chip price index: 219.18. Natural disaster risk: 3.19. Factory shutdown risk: 2.95.</code>               |
  | <code>Electronics supply disruption Israel 2022 Russia-Ukraine / US Export Controls</code>      | <code>Semiconductor disruption: Russia-Ukraine / US Export Controls (HIGH). Year: 2022. Country: Israel. Company: Tower Semiconductor. Supply disruption index: 6.42. Export control level: 4.90. Chip price index: 219.18. Natural disaster risk: 3.19. Factory shutdown risk: 2.95.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | supply_chain_ir_cosine_ndcg@10 |
|:-----:|:----:|:------------------------------:|
| -1    | -1   | 0.7897                         |
| 1.0   | 9    | 0.7849                         |
| 2.0   | 18   | 0.7902                         |
| 3.0   | 27   | 0.7959                         |


### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.1.1
- Transformers: 4.57.6
- PyTorch: 2.11.0+cpu
- Accelerate: 1.13.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->