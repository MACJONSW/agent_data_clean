# Agent Post-Training Data Pipeline

独立于 `data-juicer` 仓库的 Agent 后训练数据清洗工程，代码全部位于根目录，能力层直接复用本地 `data-juicer`，不修改其源码。

快速按 `config.yaml` 复用更多 Data-Juicer 能力，见：

- [docs/data_juicer_reuse_config_guide.md](/home/whr/pp/main/agent_data_clean/docs/data_juicer_reuse_config_guide.md)
- [examples/agent_posttrain_pipeline/config.data_juicer_reuse.yaml](/home/whr/pp/main/agent_data_clean/examples/agent_posttrain_pipeline/config.data_juicer_reuse.yaml)

## 复用 Data-Juicer 的能力

- 多格式接入：直接复用 Data-Juicer 的 `JSONL / CSV / Parquet` 加载器
- 近重复去重：直接复用 `DocumentMinhashDeduplicator`
- token 统计：配置 `quality.hf_tokenizer` 时直接复用 `TokenNumFilter`
- 导出：直接复用 `Exporter`
- `ray_vllm` 推理：直接复用 `LLMRayVLLMEnginePipeline`
- `Arrow / Feather`：Data-Juicer 未直接提供 formatter，因此仅在本项目补了外部加载器

## 算子化扩展

- `n-gram` 去污染现在已经抽成项目内 operator：`agent_posttrain_pipeline/operators.py::NGramContaminationOperator`
- 其余 Data-Juicer 算子可以通过配置直接注入，无需改主流程代码
- 适合快速复用的前提：算子主要读写标准化后的 `text / stats / metadata` 字段

示例：

```yaml
operators:
  - hook: post_quality
    name: alnum_gate
    class_path: data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter
    init_kwargs:
      text_key: text
      min_ratio: 0.5
```

支持的注入位置：

- `post_normalized`
- `post_dedup`
- `post_contamination`
- `post_quality`
- `post_augmentation`
- `post_judge`
- `pre_export`

## 企业版目录

- `agent_posttrain_pipeline/`: 核心业务代码
- `configs/`: `base / dev / prod / ray_vllm` 配置分层
- `scripts/`: 运行、校验、CI smoke 脚本
- `monitoring/`: 指标导出、Prometheus 告警、Grafana dashboard、日志配置
- `tests/agent_posttrain_pipeline/`: 单元测试
- `.github/workflows/ci.yml`: CI 工作流
- `Dockerfile`: 容器化构建
- `Makefile`: 常用工程命令

## 运行

1. 本地示例：

```bash
python run_agent_posttrain_pipeline.py --config examples/agent_posttrain_pipeline/config.yaml
```

2. 企业化目录配置：

```bash
python run_agent_posttrain_pipeline.py --config configs/dev.yaml
```

3. Shell 包装脚本：

```bash
bash scripts/run_pipeline.sh configs/prod.yaml
```

## Makefile

```bash
make pycompile
make test
make validate-example
make run-example
make smoke
make docker-build
```

## 监控输出

运行完成后会在工作目录下输出：

- `monitoring/pipeline_metrics.json`
- `monitoring/pipeline_metrics.prom`
- `visualizations/pipeline.svg`
- `visualizations/report.html`

可直接接入：

- `monitoring/prometheus/alerts.yml`
- `monitoring/grafana/agent_pipeline_dashboard.json`

## CI

CI 文件位于 `.github/workflows/ci.yml`，默认执行：

- 本地 `data-juicer` 安装
- 根项目安装
- `make pycompile`
- `make test`
- `make validate-example`
