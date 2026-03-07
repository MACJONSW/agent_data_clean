# 本工程 vs Data-Juicer 职责边界图

```mermaid
flowchart LR
    U[用户 / 调度系统\nCLI / Makefile / CI / Docker] --> C[本工程配置层\nconfigs/*\nscripts/*]
    C --> P[本工程编排层\nAgentDataPipeline]

    subgraph OWN[本工程负责]
        direction TB
        P --> A[Agent Schema / Adapter\n字段映射\nmessages/text/metadata 统一]
        P --> NG[n-gram 去污染\n参考集污染过滤]
        P --> INF[服务化推理封装\nasync_http / callable]
        P --> MON[监控与可视化\nPrometheus textfile\nGrafana/HTML/SVG]
        P --> ENG[工程化资产\nconfigs / scripts / CI / Docker]
    end

    subgraph DJ[Data-Juicer 负责]
        direction TB
        LD[格式加载器\nJsonFormatter / CsvFormatter / ParquetFormatter / load_formatter]
        DS[NestedDataset\nmap/filter/shard/export 兼容容器]
        DD[DocumentMinhashDeduplicator\n近重复去重]
        TK[TokenNumFilter\nToken 统计]
        RV[LLMRayVLLMEnginePipeline\nRay + vLLM 推理]
        EX[Exporter\nJSONL / Parquet / 分片导出]
    end

    A --> LD
    LD --> DS
    P --> DS
    P --> DD
    P --> TK
    P --> RV
    P --> EX

    RV --> M1[本地 Ray / vLLM 集群]
    INF --> M2[OpenAI-Compatible API / 内部模型服务]
    EX --> S1[对象存储 / 本地文件系统]
    MON --> S2[Prometheus / Grafana / 观测平台]

    classDef own fill:#fff7e6,stroke:#b86b00,color:#4a2a00,stroke-width:1.5px;
    classDef dj fill:#edf6ff,stroke:#1769aa,color:#0d3b66,stroke-width:1.5px;
    classDef ext fill:#f6f6f6,stroke:#666,color:#222,stroke-width:1px;

    class P,A,NG,INF,MON,ENG,OWN own;
    class LD,DS,DD,TK,RV,EX,DJ dj;
    class U,C,M1,M2,S1,S2 ext;
```

## 一句话划分

- 本工程：负责 Agent 训练数据的业务语义、企业化运行方式和工程交付。
- Data-Juicer：负责通用数据处理底座能力，能直接复用的算子和组件都直接调用，不重复实现。

## 边界解释

### 本工程负责

- Agent 数据 schema 设计与 adapter 映射
- n-gram 污染过滤
- 非 Ray 场景下的服务化推理封装
- 运行时监控、Prometheus 指标、Grafana 资产、HTML/SVG 可视化
- 配置分层、脚本、Docker、CI、部署骨架

### Data-Juicer 负责

- 多格式数据加载基础设施
- 统一 Dataset 容器和 map/filter/shard 处理范式
- MinHash 近重复去重
- 基于 tokenizer 的 token 统计
- Ray + vLLM 的批量推理 pipeline
- 通用导出能力

## 当前代码对应关系

- 本工程编排入口：`agent_posttrain_pipeline/pipeline.py`
- 本工程 adapter：`agent_posttrain_pipeline/adapters.py`
- 本工程去污染：`agent_posttrain_pipeline/contamination.py`
- 本工程监控：`monitoring/pipeline_monitor.py`
- Data-Juicer 复用桥接：`agent_posttrain_pipeline/dj_bridge.py`
