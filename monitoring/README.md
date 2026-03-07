# Monitoring Bundle

该目录提供 Agent 数据清洗 pipeline 的上线监控资产：

- `pipeline_monitor.py`: 运行期指标采集与 Prometheus textfile 导出
- `prometheus/alerts.yml`: Prometheus 告警规则样例
- `grafana/agent_pipeline_dashboard.json`: Grafana dashboard 样例
- `logging.json`: 结构化日志配置样例

运行完成后，pipeline 会在工作目录下生成 `monitoring/pipeline_metrics.json` 和 `monitoring/pipeline_metrics.prom`。
