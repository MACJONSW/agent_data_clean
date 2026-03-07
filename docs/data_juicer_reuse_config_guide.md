# 通过 `config.yaml` 快速复用 Data-Juicer 功能

这份文档只讲一件事：不改代码，只改 `config.yaml`，怎么快速切换输入格式、输出格式、Data-Juicer 算子和推理方式。

## 适用范围

- 本工程代码在根目录，`data-juicer` 作为本地依赖被直接复用
- 不修改 `data-juicer` 源码
- 优先通过配置切换能力，只有 Data-Juicer 本身没有现成功能时才补项目代码

## 快速开始

1. 复制模板配置

```bash
cp configs/base.yaml /tmp/my_agent_pipeline.yaml
```

2. 修改 `sources`、`export`、`operators`

3. 校验配置

```bash
python scripts/validate_config.py --config /tmp/my_agent_pipeline.yaml
```

4. 运行

```bash
python run_agent_posttrain_pipeline.py --config /tmp/my_agent_pipeline.yaml
```

## 哪些能力直接复用了 Data-Juicer

- 读文件：`JSONL / CSV / Parquet`
- 近重复去重：`DocumentMinhashDeduplicator`
- token 统计：`TokenNumFilter`
- 导出：`Exporter`
- `Ray + vLLM` 推理：`LLMRayVLLMEnginePipeline`
- 额外 DJ 算子注入：通过 `operators[].class_path`

说明：

- `Arrow / Feather` 读取也支持，但这是本工程补的 loader，不是 Data-Juicer 原生 formatter。
- `n-gram` 去污染现在已经被抽成项目内 operator，可和 DJ 算子一起编排。

## 1. 快速切换输入文件类型

### 方式 A：按文件后缀自动推断

```yaml
sources:
  - path: /data/agent_train.jsonl
    adapter:
      source_name: agent_jsonl
      messages_key: messages

  - path: /data/agent_train.csv
    adapter:
      source_name: agent_csv
      prompt_key: prompt
      response_key: response

  - path: /data/agent_train.parquet
    adapter:
      source_name: agent_parquet
      text_key: text
```

### 方式 B：显式指定 `format`

当后缀不标准，或者你想强制指定格式时，写 `format`：

```yaml
sources:
  - path: /data/train.data
    format: csv
    adapter:
      source_name: forced_csv
      prompt_key: prompt
      response_key: answer

  - path: /data/train.arrow
    format: arrow
    adapter:
      source_name: arrow_source
      text_key: text
```

当前支持：

- `json`
- `jsonl`
- `csv`
- `parquet`
- `arrow`
- `feather`

## 2. 快速切换字段映射

所有输入源都会先过 `adapter`，统一成标准 schema。

标准字段：

- `sample_id`
- `messages`
- `text`
- `metadata`
- `quality_signals`

常见映射方式：

### 对话样本

```yaml
sources:
  - path: /data/dialog.jsonl
    adapter:
      source_name: dialog_data
      id_key: id
      messages_key: conversations
      message_role_key: from
      message_content_key: value
```

### 指令数据

```yaml
sources:
  - path: /data/sft.csv
    adapter:
      source_name: sft_data
      instruction_key: instruction
      input_key: input
      output_key: output
```

### 问答数据

```yaml
sources:
  - path: /data/qa.parquet
    adapter:
      source_name: qa_data
      prompt_key: question
      response_key: answer
```

## 3. 快速切换输出文件类型

最终导出由 Data-Juicer `Exporter` 负责。

### 按 `export.path` 后缀自动推断

```yaml
export:
  path: /data/output/cleaned_agent_data.jsonl
  export_type: null
```

```yaml
export:
  path: /data/output/cleaned_agent_data.parquet
  export_type: null
```

### 显式指定 `export_type`

```yaml
export:
  path: /data/output/cleaned_agent_data.bin
  export_type: parquet
```

当前支持：

- `jsonl`
- `json`
- `parquet`

### 控制分片格式

中间 shard 的格式单独由 `sharding.export_type` 控制：

```yaml
sharding:
  enabled: true
  rows_per_shard: 50000
  export_type: jsonl
```

## 4. 快速切换内置算子

本工程的主流程是固定阶段，但每一段都能通过配置启停和调参。

### 去重

这是直接复用的 Data-Juicer `DocumentMinhashDeduplicator`：

```yaml
dedup:
  enabled: true
  tokenization: character
  window_size: 5
  lowercase: true
  num_permutations: 256
  jaccard_threshold: 0.85
```

### `n-gram` 去污染

这是本工程 operator：

```yaml
contamination:
  enabled: true
  reference_sources:
    - path: /data/eval_reference.jsonl
      adapter:
        source_name: eval_ref
        text_key: text
  ngram_size: 13
  max_contamination_ratio: 0.2
  min_matched_ngrams: 2
```

### token 统计和质量过滤

配置 `hf_tokenizer` 时，会直接复用 Data-Juicer `TokenNumFilter` 统计 token：

```yaml
quality:
  enabled: true
  hf_tokenizer: Qwen/Qwen2.5-7B-Instruct
  min_tokens: 32
  max_tokens: 8192
  min_turns: 2
  min_assistant_turns: 1
```

## 5. 快速注入别的 Data-Juicer 算子

这是快速复用 Data-Juicer 其余功能的核心方式。

### 配置格式

```yaml
operators:
  - hook: post_quality
    name: alnum_gate
    class_path: data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter
    init_kwargs:
      text_key: text
      min_ratio: 0.5
```

字段含义：

- `hook`: 把算子挂到哪一个阶段位置
- `name`: 日志和可视化里的显示名
- `class_path`: Python import path
- `init_kwargs`: 算子构造参数
- `run_kwargs`: 调用 `run(dataset, ...)` 时的额外参数

### 支持的 hook

- `post_normalized`
- `post_dedup`
- `post_contamination`
- `post_quality`
- `post_augmentation`
- `post_judge`
- `pre_export`

这些 hook 现在是“阶段位置 hook”：

- 即使你把内置 `dedup` 关掉，`post_dedup` 位置仍然可挂算子
- 即使你把内置 `quality` 关掉，`post_quality` 位置仍然可挂算子

### 常见用法

#### 在质量过滤后加一个 Data-Juicer 字母数字比例过滤器

```yaml
operators:
  - hook: post_quality
    name: alnum_gate
    class_path: data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter
    init_kwargs:
      text_key: text
      min_ratio: 0.6
```

#### 用别的 DJ 去重器替换默认去重

思路：

- 先把内置 `dedup.enabled` 关掉
- 再在 `post_normalized` 或 `post_dedup` 注入你想要的 DJ 去重器

```yaml
dedup:
  enabled: false

operators:
  - hook: post_dedup
    name: custom_dedup
    class_path: your_package.your_dj_style_deduplicator.CustomDeduplicator
    init_kwargs:
      text_key: text
```

#### 在导出前加一个自定义 mapper

```yaml
operators:
  - hook: pre_export
    name: redact_fields
    class_path: my_project.ops.RedactMetaMapper
    init_kwargs:
      text_key: text
```

## 6. 快速切换模型推理后端

### 服务化接口

```yaml
augmentation:
  enabled: true
  backend: async_http
  base_url: http://127.0.0.1:8000/v1
  model: your-model
  batch_size: 32
  max_concurrency: 16
```

### Data-Juicer `ray_vllm`

```yaml
augmentation:
  enabled: true
  backend: ray_vllm
  model: /models/YourModel
  batch_size: 64
  sampling_params:
    temperature: 0.7
    max_tokens: 1024
```

`judge` 的配置方式和 `augmentation` 相同。

## 7. 推荐的配置改法

如果你想快速切换实验，不建议直接改 `configs/base.yaml`。

推荐做法：

1. 复制一份模板
2. 只改这一份业务 `config.yaml`
3. 用 `validate_config.py` 先校验
4. 再运行 pipeline

示例：

```bash
cp examples/agent_posttrain_pipeline/config.data_juicer_reuse.yaml /tmp/experiment.yaml
python scripts/validate_config.py --config /tmp/experiment.yaml
python run_agent_posttrain_pipeline.py --config /tmp/experiment.yaml
```

## 8. 什么时候不能只靠配置

下面几类情况通常需要补代码：

- Data-Juicer 没有现成 loader，且项目里也没有对应实现
- 目标算子依赖多模态字段，但当前标准 schema 只有文本字段
- 目标算子依赖特殊字段名，而 adapter 还没有把源数据对齐到这些字段
- 你需要配置继承、配置合并、远端配置中心等更重的配置系统

## 9. 最小可运行模板

可直接参考：

- 示例模板：[config.data_juicer_reuse.yaml](/home/whr/pp/main/agent_data_clean/examples/agent_posttrain_pipeline/config.data_juicer_reuse.yaml)
- 基础模板：[base.yaml](/home/whr/pp/main/agent_data_clean/configs/base.yaml)
- 主 README：[README.md](/home/whr/pp/main/agent_data_clean/README.md)
