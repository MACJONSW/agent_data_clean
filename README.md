# LLM 后训练数据清洗 Pipeline 功能复现手册

本文不是代码设计分析，而是面向“从 0 开始复现”的功能说明书。重点说明：
- 每个功能块解决什么问题。
- 输入目录里应该放什么格式的数据。
- 中间数据长什么样。
- 输出目录会产出什么。
- 关键配置字段怎么写。
- 哪些地方可以自定义，例如 `adapter`、`query_builder`、`formatter`。

---

## 1. 先理解这套系统的最小运行模型

这套系统的核心不是“某一个脚本”，而是：
- 一个 `pipeline YAML`
- 一组 `block`
- 一个统一的中间数据格式

执行入口是 [run_blocks.py](/home/whr/pp/main/llm_data/run_blocks.py)。

最小 pipeline 长这样：

```yaml
name: demo

runtime:
  input_dir: ./data/raw
  output_dir: ./output/demo
  log_path: ./logs/demo
  tasks: 8
  workers: 8

blocks:
  - id: chunk_input
    type: dataset_chunker
    config:
      chunk_size: 1000

  - id: token_count
    type: token_counter
    config:
      response_key: answer
      tokenizer:
        name_or_path: /path/to/tokenizer

  - id: token_filter
    type: numeric_filter
    config:
      metadata_key: total_tokens
      filter:
        threshold: 4096
        comparison: "<="
        skip_missing: true
```

运行时每个 block 都会自动接收上一个 block 的输出目录作为输入目录，除非你显式指定 `config.input_dir`。

---

## 2. 统一中间数据格式

几乎所有清洗、推理、过滤 block 的输入，最终都要变成 Datatrove 的 `Document` 语义，对应 JSONL 里通常长这样：

```json
{
  "text": "用户问题或主文本",
  "id": "sample-001",
  "metadata": {
    "answer": "参考答案或模型回答",
    "ground_truth": "标准答案",
    "data_source": "dataset_name"
  }
}
```

字段约定：
- `text`: 主文本。大多数 block 都默认从这里读取 prompt / instruction。
- `id`: 唯一 ID。建议全局唯一。
- `metadata`: 附加字段。后续 block 会把统计值、分数、推理结果都写在这里。

最重要的结论：
- **原始数据可以是任意格式。**
- **一旦进入 pipeline 主链，最好尽快统一成 `text + id + metadata`。**

---

## 3. 运行时变量和 block 输出引用

每个 block 执行后，builder 会把输出目录注入到 runtime 变量里。

规则：
- `${block_id_output}`: 这个 block 的主输出目录
- `${block_id_<named_output>}`: 这个 block 的命名输出目录

例如：
- `token_filter` 的主输出变量是 `${token_filter_output}`
- `numeric_filter` 还会额外产出 `${token_filter_stats}` 和 `${token_filter_percentiles}`
- `ngrams_decont` 还会额外产出 `${ngram_decont_index}` 和 `${ngram_decont_excluded}`

这意味着后一个 block 可以直接这样写：

```yaml
input_dir: ${token_filter_output}
```

---

## 4. 三类最重要的自定义接口

## 4.1 Reader Adapter

作用：把原始记录映射成统一中间格式。

函数签名：

```python
def my_adapter(reader, data: dict, path: str, id_in_file: int | str) -> dict:
    ...
```

返回值至少要包含：
- `text`
- `id`

通常还会返回：
- `metadata`

示例：

```python
def my_adapter(_reader, data, path, id_in_file):
    return {
        "text": data["prompt"],
        "id": f"{path}:{id_in_file}",
        "metadata": {
            "answer": data["response"],
            "source": path,
        },
    }
```

配置方式：

```yaml
adapter: mypkg.adapters.reader:openmath2_request_adapter
```

现有示例可参考 [reader.py](/home/whr/pp/main/llm_data/src/mypkg/adapters/reader.py)。

---

## 4.2 Writer Adapter

作用：控制最终写盘 JSON 的字段结构。

函数签名：

```python
def my_writer_adapter(writer, document) -> dict:
    ...
```

常见用途：
- 写盘前删除 `metadata.inference_results`
- 保留某些评分字段
- 改写输出字段名

示例：

```python
def remove_inference_results_writer_adapter(writer, document):
    data = writer._default_adapter(document)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("inference_results", None)
    return data
```

现有示例可参考 [writer.py](/home/whr/pp/main/llm_data/src/mypkg/adapters/writer.py)。

---

## 4.3 Query Builder

作用：把一个 `Document` 变成模型请求 payload。

函数签名：

```python
def my_query_builder(runner, document) -> dict | list[dict]:
    ...
```

如果 `use_chat=true`，返回值应包含：
- `messages`

例如：

```python
def math_query_builder(_runner, document):
    return {
        "messages": [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": document.text},
        ],
        "max_tokens": 16384,
    }
```

如果 `use_chat=false`，返回值应包含：
- `prompt`

现有示例可参考 [query_builder.py](/home/whr/pp/main/llm_data/src/mypkg/query_builder.py)。

---

## 5. 功能 1：读取原始数据并标准化成统一 JSONL

对应 block：`dataset_chunker`

代码位置：
- [dataset_chunker.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/dataset_chunker.py)
- [dataset_chunker.yaml](/home/whr/pp/main/llm_data/src/lego_pipeline/defaults/dataset_chunker.yaml)

### 5.1 解决什么问题

把多种原始格式：
- JSONL
- Parquet
- CSV
- Arrow / IPC

统一读取出来，经过 adapter 转成标准 `Document`，然后切成固定大小的 JSONL shard。

### 5.2 输入形式

`input_dir` 可以是：
- 单个目录
- 多个目录的逗号拼接字符串
- 多个目录的列表

目录中可以混合存在：
- `*.jsonl`
- `*.jsonl.gz`
- `*.jsonl.zst`
- `*.parquet`
- `*.csv`
- `*.arrow`

如果你传的是原始数据，而不是已经标准化的中间 JSONL，建议一定配 `adapter`。

### 5.3 Adapter 输入和输出

`dataset_chunker` 的 reader 会先读原始记录，再调用 adapter。

例如原始 OpenMath 样本：

```json
{
  "messages": [
    {"role": "user", "content": "Compute 1+1"},
    {"role": "assistant", "content": "The answer is \\boxed{2}"}
  ]
}
```

经过 `openmath2_request_adapter` 后会变成：

```json
{
  "text": "Compute 1+1",
  "id": 0,
  "metadata": {
    "answer": "The answer is \\boxed{2}",
    "ground_truth": "2",
    "data_source": "ai2-adapt-dev/openmath-2-math"
  }
}
```

### 5.4 关键配置

```yaml
- id: chunk_input
  type: dataset_chunker
  config:
    adapter: mypkg.adapters.reader:openmath2_request_adapter
    chunk_size: 1000
    compression: gzip
    jsonl_reader:
      enabled: true
    parquet_reader:
      enabled: true
    csv_reader:
      enabled: true
    ipc_reader:
      enabled: true
    execution:
      type: local
      tasks: 1
      workers: 1
```

重点字段：
- `chunk_size`: 每个输出分片包含多少条样本
- `compression`: 输出压缩，可取 `gzip` / `zstd` / `none` / `infer`
- `<format>_reader.enabled`: 开关不同格式 reader
- `<format>_reader.glob_pattern`: 自定义匹配模式

### 5.5 输出形式

输出目录是：
- `${output_dir}/chunk_input`，如果没有单独覆写 `output_dir`

默认文件名模式：

```text
chunk_${rank}_${chunk_index}.jsonl.gz
```

例如：

```text
output/demo/chunk_input/chunk_0_00000.jsonl.gz
output/demo/chunk_input/chunk_0_00001.jsonl.gz
```

每一行都是标准中间格式 JSON：

```json
{
  "text": "Compute 1+1",
  "id": "0",
  "metadata": {
    "answer": "The answer is \\boxed{2}",
    "ground_truth": "2"
  }
}
```

### 5.6 这个功能的特点

- 支持多源目录输入
- 支持多格式混读
- adapter 是标准化入口，几乎所有自定义都从这里开始
- 产出的是后续 block 最容易消费的统一 JSONL

---

## 6. 功能 2：近重复去重

对应 block：`minhash`

代码位置：
- [minhash.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/minhash.py)

### 6.1 解决什么问题

对已经标准化的样本进行近重复去重。

### 6.2 输入形式

输入目录应该是标准中间 JSONL，例如：

```json
{"text": "question A", "id": "1", "metadata": {"answer": "..."}} 
{"text": "question A", "id": "2", "metadata": {"answer": "..."}} 
```

`minhash` 只要求样本能被 `JsonlReader` 正常读成 `Document`，重点字段是：
- `text`
- `id`

### 6.3 关键配置

```yaml
- id: minhash
  type: minhash
  config:
    language: english
    minhash:
      n_grams: 5
      num_buckets: 10
      hashes_per_bucket: 8
      seed: 1
      precision: 64
      hash_fc: xxhash
    execution:
      type: local
      tasks: 8
      workers: 8
```

参数含义：
- `n_grams`: 文本切分粒度
- `num_buckets` / `hashes_per_bucket`: MinHash 召回与计算成本的平衡参数
- `language`: 语言相关切分策略

### 6.4 输出形式

主输出：
- `${minhash_output}`: 去重后的 JSONL 目录

中间目录：
- `signatures`
- `buckets`
- `remove_ids`

如果不手动指定，默认在：

```text
{output_dir}_intermediate/minhash/
```

主输出里的每一行仍然保持原样本结构，只是重复样本被移除了。

### 6.5 这个功能的特点

- 不改变样本 schema
- 只做“删除重复样本”
- 适合作为所有清洗链路的前置步骤

---

## 7. 功能 3：n-gram 去污染

对应 block：`ngrams_decont`

代码位置：
- [ngrams_decont.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/ngrams_decont.py)

### 7.1 解决什么问题

把目标数据与“参考语料/评测集索引”做 n-gram 对比，过滤掉污染样本。

### 7.2 输入形式

主输入：
- `input_dir`: 要清洗的数据，标准中间 JSONL

索引参考输入：
- `index.input_dir`: 参考语料目录，用来构建污染索引

### 7.3 关键配置

```yaml
- id: decont
  type: ngrams_decont
  config:
    index:
      build: true
      input_dir: /path/to/reference_dataset
    decont_config:
      n_grams: 8
    exclusion_writer:
      output_folder: ${output_dir}/contaminated
    execution:
      type: local
      tasks: 8
      workers: 8
```

### 7.4 输出形式

命名输出有三个：
- `${decont_output}`: 过滤后的样本目录
- `${decont_index}`: 构建出来的索引目录
- `${decont_excluded}`: 被判定为污染的样本目录

主输出仍然是标准中间 JSONL。

### 7.5 这个功能的特点

- 它不是“内部相似样本去重”，而是“和参考集比对”
- 适合训练集对评测集去污染
- 可单独保留被污染样本用于审计

---

## 8. 功能 4：统计 token 长度

对应 block：`token_counter`

代码位置：
- [token_counter.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/token_counter.py)
- [formatted_counter.py](/home/whr/pp/main/llm_data/src/operators/tokens/formatted_counter.py)

### 8.1 解决什么问题

根据 prompt 模板和 response 字段，计算：
- `prompt_tokens`
- `response_tokens`
- `total_tokens`

### 8.2 输入形式

输入必须至少具备：
- `text`
- `metadata[response_key]`

例如：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2"
  }
}
```

如果 `response_key: answer`，那么它会取 `metadata.answer` 作为回答。

### 8.3 关键配置

```yaml
- id: token_count
  type: token_counter
  config:
    system_prompt: You are a helpful assistant.
    response_key: answer
    tokenizer:
      name_or_path: /path/to/tokenizer
    prompt_template: null
    full_template: null
    count_eos_token: false
```

说明：
- `response_key`: 取哪个 metadata 字段当作回答
- `tokenizer.name_or_path`: 必填
- `prompt_template`: 只算 prompt 时的模板
- `full_template`: 算完整对话时的模板

默认模板是 ChatML 风格：

```text
system + user + assistant
```

### 8.4 输出形式

输出目录是标准 JSONL，只是 `metadata` 里新增 3 个字段：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2",
    "prompt_tokens": 18,
    "response_tokens": 3,
    "total_tokens": 21
  }
}
```

### 8.5 这个功能的特点

- 不改变原始 `text` 和 `answer`
- 只做 metadata 增量写回
- 是长度过滤的前置步骤

---

## 9. 功能 5：按数值字段过滤样本

对应 block：`numeric_filter`

代码位置：
- [numeric_filter.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/numeric_filter.py)
- [metadata_stats.py](/home/whr/pp/main/llm_data/src/operators/stats/metadata_stats.py)

### 9.1 解决什么问题

按某个 metadata 数值字段过滤样本，比如：
- `total_tokens <= 4096`
- `judge_score >= 4.0`
- `difficulty >= 0.7`

### 9.2 输入形式

输入样本必须满足：
- 存在 `metadata_key` 对应的字段
- 该字段可转换为数字

例如：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2",
    "total_tokens": 21
  }
}
```

`metadata_key` 既可以是字符串，也可以是嵌套路径列表，例如：

```yaml
metadata_key: total_tokens
```

或：

```yaml
metadata_key: [judge_score, correctness]
```

### 9.3 关键配置

```yaml
- id: token_filter
  type: numeric_filter
  config:
    metadata_key: total_tokens
    filter:
      threshold: 4096
      comparison: "<="
      skip_missing: true
      keep_missing: false
    percentiles: [50, 75, 90, 95, 99]
```

比较符支持：
- `>=`
- `>`
- `<=`
- `<`
- `==`
- `!=`

你也可以不用 `threshold`，改用 `percentile`：

```yaml
filter:
  percentile: 95
  comparison: "<="
```

这时阈值会从统计文件里自动读。

### 9.4 输出形式

`numeric_filter` 会产生 3 类输出：

主输出：
- `${token_filter_output}`: 过滤后的样本目录

统计输出：
- `${token_filter_stats}`: 统计原始中间目录
- `${token_filter_percentiles}`: 百分位结果目录

主输出 JSONL 仍然保持原样本结构，只保留通过过滤的样本。

百分位文件示例：

```json
{
  "metadata_key": "total_tokens",
  "total_documents": 10000,
  "min": 12,
  "max": 8192,
  "mean": 1260.4,
  "percentiles": {
    "p50": 980,
    "p90": 2400,
    "p95": 3200,
    "p99": 6000
  }
}
```

### 9.5 这个功能的特点

- 先统计，后过滤
- 既能做硬阈值过滤，也能做分位点过滤
- 非常适合配合 `token_counter` 和 `llm_as_judge`

---

## 10. 功能 6：通用处理链

对应 block：`operator_chain`

代码位置：
- [operator_chain.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/operator_chain.py)

### 10.1 解决什么问题

不新增 Python block，直接在 YAML 里拼：
- reader
- steps
- writer

适合实现轻量自定义流程，例如：
- 样本抽样
- lambda 过滤
- 自定义写盘
- SFT 导出

### 10.2 输入形式

取决于你的 `reader`。

最常见情况：
- 如果不写 `reader`，默认读上一个 block 输出的标准 JSONL
- 如果显式写 `reader.import: datatrove.pipeline.readers.JsonlReader`，同样是读标准 JSONL

### 10.3 配置格式

```yaml
- id: sample_num_filter
  type: operator_chain
  config:
    reader:
      import: datatrove.pipeline.readers.JsonlReader
    steps:
      - import: operators.filter.sample_num_filter:SampleNumFilter
        params:
          num_keep: 5000
          seed: 1234
    writer:
      import: datatrove.pipeline.writers.JsonlWriter
    execution:
      type: local
      tasks: 1
```

### 10.4 支持的两种参数导入方式

直接导入函数/对象：

```yaml
filter_function: import:mypkg.filters:math_incorrect_filter
```

实例化类对象：

```yaml
exclusion_writer:
  class: import:datatrove.pipeline.writers.jsonl.JsonlWriter
  params:
    output_folder: ${output_dir}/excluded
    compression: gzip
```

### 10.5 输出形式

如果配置了 writer，则输出目录由 writer 决定。

如果没有配置 writer，但 step 本身带 `output_folder`，则 `operator_chain` 会把那个目录当成输出目录。

这使得 `operator_chain` 非常适合承载“导出类 step”。

### 10.6 这个功能的特点

- 是“万能拼装块”
- 适合快速接 Datatrove 算子
- 复现时非常常用，因为无需新写 block

---

## 11. 功能 7：批量模型推理

对应 block：`async_model_serve`

代码位置：
- [async_model_serve.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/async_model_serve.py)
- [async_model_server.py](/home/whr/pp/main/llm_data/src/operators/llm_query/async_model_server.py)

### 11.1 解决什么问题

对输入样本批量调用 vLLM/OpenAI 兼容接口，写回模型生成结果。

### 11.2 输入形式

输入样本通常仍然是标准中间 JSONL：

```json
{
  "text": "Solve x+1=2",
  "id": "1",
  "metadata": {
    "ground_truth": "1"
  }
}
```

然后通过 `query_builder` 把这个样本转换成请求。

### 11.3 Query Builder 输出格式

如果 `use_chat: true`，典型输出：

```python
{
    "messages": [
        {"role": "system", "content": "Please reason step by step."},
        {"role": "user", "content": document.text},
    ],
    "max_tokens": 16384,
}
```

如果 `use_chat: false`，则应返回：

```python
{
    "prompt": "...",
    "max_tokens": 1,
}
```

### 11.4 关键配置

```yaml
- id: qwen_query
  type: async_model_serve
  config:
    query_builder: mypkg.query_builder:math_query_builder
    sampling:
      temperature: 0.6
      top_p: 0.95
      max_tokens: 4096
    model_server:
      result_key: inference_results
      sample_count: 1
    server:
      model_path: /path/to/model
      tensor_parallel_size: 2
      gpus_per_node: 8
      wait_timeout: 1800
    client:
      max_concurrent_requests: 512
      max_concurrent_tasks: 1024
      request_timeout: 1800
      max_retries: 4
```

重点字段：
- `query_builder`: 最核心，决定请求长什么样
- `sampling`: 生成参数默认值
- `model_server.result_key`: 结果写回哪个 metadata 字段
- `model_server.sample_count`: 每条输入采样几次

### 11.5 输出形式

输出仍然是标准 JSONL，但 `metadata` 中会新增 `result_key` 对应字段。

默认是：

```json
{
  "text": "Solve x+1=2",
  "id": "1",
  "metadata": {
    "ground_truth": "1",
    "inference_results": [
      {
        "text": "We have x=1. Therefore the final answer is: \\boxed{1}.",
        "finish_reason": "stop",
        "usage": {
          "prompt_tokens": 120,
          "completion_tokens": 180,
          "total_tokens": 300
        },
        "raw_choice": {
          "...": "原始模型返回 choice"
        }
      }
    ]
  }
}
```

如果模型返回了 `logprobs`，结果项里还会包含：
- `logprobs`

如果 `sample_count > 1`，`inference_results` 会是多个结果项组成的列表。

### 11.6 默认输出文件

默认写成：

```text
${rank}_chunk_${chunk_index}.jsonl.gz
```

### 11.7 这个功能的特点

- 保留原样本，不覆盖原字段
- 推理结果统一写进 `metadata[result_key]`
- 非常适合做蒸馏、生成补全、评审前回答生成

---

## 12. 功能 8：LLM 自动判分

对应 block：`llm_as_judge`

代码位置：
- [llm_as_judge.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/llm_as_judge.py)
- [llm_as_judge.yaml](/home/whr/pp/main/llm_data/src/lego_pipeline/defaults/llm_as_judge.yaml)

### 12.1 解决什么问题

把一个样本送给 Judge 模型，根据 prompt 模板输出分数或多维评分。

### 12.2 输入形式

Judge 有两种 prompt 形态：

`Q_*`：
- 只用问题
- 需要 `text`

`QA_*`：
- 问题 + 回答
- 需要 `text` 和 `response_key` 对应字段

例如输入样本：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2"
  }
}
```

如果：

```yaml
prompt:
  name: QA_Correctness.txt
response_key: answer
```

那么 Judge 模型收到的 prompt 中会同时包含：
- `{instruction}` -> `text`
- `{output}` -> `metadata.answer`

### 12.3 关键配置

```yaml
- id: judge
  type: llm_as_judge
  config:
    prompt:
      name: QA_Correctness.txt
    response_key: answer
    save_reason: true
    model_server:
      result_key: judge_score
    server:
      model_path: /path/to/judge_model
```

### 12.4 输出形式

`llm_as_judge` 底层先把原始模型回复放到：
- `judge_score_raw`

然后 post-process 会把它解析后再写成最终字段，并把 raw 字段移除。

单分数场景最终输出：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2",
    "judge_score": 4.5,
    "judge_score_reason": [
      "{\"score\": 4.5, \"reason\": \"Correct and concise\"}"
    ]
  }
}
```

多维评分场景最终输出可能是：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "answer": "2",
    "judge_score": {
      "clarity": 4.0,
      "correctness": 5.0,
      "completeness": 4.0
    }
  }
}
```

### 12.5 这个功能的特点

- 和 `async_model_serve` 共用同一套服务能力
- 差别在于内置了 prompt 解析和结果后处理
- 输出天然适合交给 `numeric_filter` 再过滤

---

## 13. 功能 9：把模型输出或现成答案导出成 SFT 数据

对应实现：
- `operator_chain + SFTExportStep`

代码位置：
- [sft_export.py](/home/whr/pp/main/llm_data/src/operators/export/sft_export.py)
- [exporter.py](/home/whr/pp/main/llm_data/src/mypkg/exporter.py)

### 13.1 解决什么问题

把中间 JSONL 变成训练可直接消费的：
- JSONL SFT 数据
- Parquet SFT 数据

### 13.2 两种常见输入来源

第一种：来自 `async_model_serve`

这时默认 formatter 会读取：
- `metadata.inference_results[*].text`

输入示例：

```json
{
  "text": "What is 1+1?",
  "id": "1",
  "metadata": {
    "inference_results": [
      {"text": "2"}
    ]
  }
}
```

第二种：来自直接 SFT 数据集

这时通常不用默认 formatter，而是使用：

```yaml
formatter: mypkg.exporter:process_opendata_sft_formatter
```

它会读取：
- `text`
- `metadata.answer`

### 13.3 配置方式

```yaml
- id: sft_export
  type: operator_chain
  config:
    enable_default_writer: false
    reader:
      import: datatrove.pipeline.readers.JsonlReader
    steps:
      - import: operators.export.sft_export:SFTExportStep
        params:
          output_folder: ${output_dir}/sft_data
          output_format: parquet
          chunk_size: 20000
          formatter: mypkg.exporter:process_opendata_sft_formatter
          formatter_kwargs:
            system_prompt: You are a helpful assistant.
    execution:
      type: local
      tasks: 8
      workers: 8
```

### 13.4 输出格式

输出记录格式：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 1+1?"},
    {"role": "assistant", "content": "2"}
  ],
  "data_source": "dataset_name",
  "extra_info": {
    "ground_truth": "2",
    "difficulty": 0.7
  }
}
```

说明：
- 默认 formatter 只会写 `messages`
- 自定义 formatter 可以附加 `data_source`、`extra_info`
- 输出格式可选 `jsonl` 或 `parquet`

### 13.5 这个功能的特点

- 是“从中间样本到训练样本”的关键一步
- 既支持模型生成结果导出，也支持现成答案导出
- formatter 决定最终训练数据 schema

---

## 14. 功能 10：复用子流程

对应 block：`block_chain`

代码位置：
- [block.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/block_chain/block.py)
- [sft_data_preprocess.yaml](/home/whr/pp/main/llm_data/pipelines/bundles/sft_data_preprocess.yaml)

### 14.1 解决什么问题

把一串常见 block 打包成 bundle 复用。

例如把：
- `dataset_chunker`
- `token_counter`
- `numeric_filter`
- `sft_export`

打成一个标准预处理 bundle。

### 14.2 输入形式

`block_chain` 自己不关心样本结构，它只负责：
- 加载子 YAML
- 注入参数
- 运行内部 blocks

### 14.3 关键配置

```yaml
- id: preprocess
  type: block_chain
  config:
    chain: sft_data_preprocess
    search_paths:
      - pipelines/bundles
    parameters:
      formatter: mypkg.exporter:process_opendata_sft_formatter
    block_overrides:
      token_filter:
        filter:
          threshold: 8192
```

### 14.4 输出形式

默认会把子流程里每个 block 的输出继续传播到外层 runtime。

例如子流程里有 `sft_export`，则外层可拿到：
- `${preprocess_sft_export_output}` 或通过最终 `output`

### 14.5 这个功能的特点

- 非常适合做标准化 pipeline 模板
- 让“复现”从写很多 block 变成只填参数

---

## 15. 功能 11：实验批量展开

对应 block：`experiment`

代码位置：
- [experiment.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/experiment.py)

### 15.1 解决什么问题

自动展开多个：
- `sample_rate`
- `model_path`

组合成多组 `train_eval` 任务。

### 15.2 输入形式

输入目录通常是已经导出的训练数据目录，通常为：
- Parquet 目录

### 15.3 配置形式

典型思想是：

```yaml
chain_config:
  data:
    sample_rates: [0.1, 0.3, 1.0]
  model:
    model_paths:
      - /path/model_a
      - /path/model_b
```

然后自动生成多个 `train_eval` block。

### 15.4 输出形式

每个组合会变成一个独立 `train_eval` 任务，输出各自的评测报告目录。

### 15.5 这个功能的特点

- 适合数据量/模型规模对比实验
- 不直接改变数据格式
- 属于编排层能力

---

## 16. 功能 12：训练 + 评测闭环

对应 block：`train_eval`

代码位置：
- [train_eval.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/train_eval.py)
- [train_eval.yaml](/home/whr/pp/main/llm_data/src/lego_pipeline/defaults/train_eval.yaml)

### 16.1 解决什么问题

把清洗和导出的训练数据直接接到训练脚本，再接评测脚本，形成闭环。

### 16.2 输入形式

`input_dir` 应该是训练数据目录，通常是：
- `sft_export` 导出的 Parquet 目录

例如：

```text
output/demo/sft_data/
  sft_rank_0_chunk_00000.parquet
  sft_rank_1_chunk_00000.parquet
```

### 16.3 关键配置

```yaml
- id: train_then_eval
  type: train_eval
  config:
    input_dir: ${sft_export_output}
    output_dir: ${output_dir}/eval_report
    train:
      project:
        project_name: SFT_Qwen2.5-3B-Instruct
        exp_name: demo_exp
      model:
        model_path: /path/to/model
      train:
        lr: 4e-6
        batch_size: 128
        micro_batch_size: 2
        epochs: 1
        max_length: 8192
    eval:
      data:
        datasets: math_500
      eval:
        judge_model_name: Qwen2.5-32B-Instruct
```

### 16.4 输出形式

这个 block 本质上是“提交训练和评测任务”，不是数据转换块。

输出主要是：
- `output_dir`: 评测报告目录

训练 checkpoint 的实际落盘逻辑由外部脚本负责，不完全由本仓库管理。

### 16.5 复现时必须满足的前置条件

需要环境变量：
- `SWANLAB_API_KEY`
- `AUTH_USER`

需要外部项目路径存在：
- `train.project.llm_train_root`
- `eval.project.openai_evals_plugin_root`

### 16.6 这个功能的特点

- 它不是纯本仓库内闭环
- 强依赖外部训练/评测脚本与平台环境
- 对“数据清洗复现”来说，它是最后一跳

---

## 17. 一条最小可复现的数据清洗到导出链路

如果你要从 0 开始跑，最建议先复现这条：

1. 原始 JSONL
2. `dataset_chunker`
3. `token_counter`
4. `numeric_filter`
5. `sft_export`

示例 YAML：

```yaml
name: minimal_sft_data_pipeline

runtime:
  input_dir: ./data/raw
  output_dir: ./output/minimal
  log_path: ./logs/minimal
  tasks: 8
  workers: 8

blocks:
  - id: chunk_input
    type: dataset_chunker
    config:
      adapter: mypkg.adapters.reader:openmath2_request_adapter
      chunk_size: 1000
      execution:
        type: local
        tasks: 1
        workers: 1

  - id: token_count
    type: token_counter
    config:
      system_prompt: You are a helpful assistant.
      response_key: answer
      tokenizer:
        name_or_path: /path/to/tokenizer
      execution:
        type: local
        tasks: 8
        workers: 8

  - id: token_filter
    type: numeric_filter
    config:
      metadata_key: total_tokens
      filter:
        threshold: 4096
        comparison: "<="
        skip_missing: true
      execution:
        type: local
        tasks: 8
        workers: 8

  - id: sft_export
    type: operator_chain
    config:
      enable_default_writer: false
      reader:
        import: datatrove.pipeline.readers.JsonlReader
      steps:
        - import: operators.export.sft_export:SFTExportStep
          params:
            output_folder: ${output_dir}/sft_data
            output_format: parquet
            chunk_size: 20000
            formatter: mypkg.exporter:process_opendata_sft_formatter
            formatter_kwargs:
              system_prompt: You are a helpful assistant.
      execution:
        type: local
        tasks: 8
        workers: 8
```

这条链跑通之后，再加：
- `minhash`
- `ngrams_decont`
- `async_model_serve`
- `llm_as_judge`

会更容易。

---

## 18. 从 0 复现时的推荐顺序

### 第一步：准备原始数据

你只需要保证原始样本能被 adapter 解释。

### 第二步：先写 adapter

这是最关键的一步。确认 adapter 输出：
- `text`
- `id`
- `metadata.answer` 或你后面要用到的字段

### 第三步：先只跑 `dataset_chunker`

检查输出 JSONL 是否已经长成统一中间格式。

### 第四步：再加 `token_counter`

确认 `metadata.total_tokens` 是否正常写入。

### 第五步：再加 `numeric_filter`

确认统计文件和过滤结果都正常。

### 第六步：最后加导出或模型推理

如果是直接 SFT：
- 用 `process_opendata_sft_formatter`

如果是模型蒸馏：
- 先 `async_model_serve`
- 再 `sft_export`

---

## 19. 最容易踩坑的地方

### 19.1 `adapter` 没把字段对齐

后面很多 block 默认假设：
- prompt 在 `text`
- 回答在 `metadata.answer` 或你配置的 `response_key`

如果这里没统一，后面会“能跑，但结果不对”。

### 19.2 `token_counter` 读不到 response

比如你回答字段在 `metadata.response`，但配置写的是：

```yaml
response_key: answer
```

那算出来的 `response_tokens` 会是 0。

### 19.3 `numeric_filter` 过滤字段不是数值

它最终会强制把字段转成 `float`。如果是字符串但不能转数字，会报错。

### 19.4 `llm_as_judge` 的 prompt 类型和数据不匹配

如果你用 `QA_*` prompt，却没有提供 `response_key` 对应字段，Judge prompt 会缺回答。

### 19.5 `SFTExportStep` 默认 formatter 只认 `inference_results`

如果你没有模型生成结果，而是原始答案在 `metadata.answer`，就必须显式指定：

```yaml
formatter: mypkg.exporter:process_opendata_sft_formatter
```

### 19.6 `train_eval` 不是纯本地自包含功能

它依赖外部训练项目和评测项目，不是只靠本仓库就能跑通。

---

## 20. 你真正需要记住的复现原则

如果只记三件事，记这三件：

1. 所有原始数据先用 adapter 统一成 `text + id + metadata`
2. 所有清洗 block 基本都只是在这个统一格式上增删样本或给 `metadata` 加字段
3. 训练导出不是自动猜字段，最终由 formatter 决定训练样本长什么样

---

## 21. 相关代码入口

核心 block：
- [dataset_chunker.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/dataset_chunker.py)
- [minhash.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/minhash.py)
- [ngrams_decont.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/ngrams_decont.py)
- [token_counter.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/token_counter.py)
- [numeric_filter.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/numeric_filter.py)
- [async_model_serve.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/async_model_serve.py)
- [llm_as_judge.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/llm_as_judge.py)
- [operator_chain.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/operator_chain.py)
- [train_eval.py](/home/whr/pp/main/llm_data/src/lego_pipeline/blocks/train_eval.py)

关键自定义点：
- [reader.py](/home/whr/pp/main/llm_data/src/mypkg/adapters/reader.py)
- [writer.py](/home/whr/pp/main/llm_data/src/mypkg/adapters/writer.py)
- [query_builder.py](/home/whr/pp/main/llm_data/src/mypkg/query_builder.py)
- [exporter.py](/home/whr/pp/main/llm_data/src/mypkg/exporter.py)
- [sft_export.py](/home/whr/pp/main/llm_data/src/operators/export/sft_export.py)

---

## 22. 结论

这套 pipeline 复现的关键，不在于记住所有 block 名称，而在于先掌握“统一中间格式”。

一旦你能稳定把原始数据变成：

```json
{
  "text": "...",
  "id": "...",
  "metadata": {
    "answer": "..."
  }
}
```

后面的功能几乎都只是：
- 去掉某些样本
- 给 `metadata` 增加新字段
- 把数据改写成训练格式

如果你需要，我下一步可以继续把这份文档补成“逐 block 可直接复制运行的 YAML 模板手册”，每个功能给你一份最小可运行配置。
