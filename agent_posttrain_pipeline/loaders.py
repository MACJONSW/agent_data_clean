from __future__ import annotations

from typing import List, Sequence

import pyarrow as pa
import pyarrow.feather as feather
from datasets import Dataset, concatenate_datasets

from .adapters import AgentSampleAdapter
from .config import DataSourceConfig
from .dj_bridge import CsvFormatter, JsonFormatter, NestedDataset, ParquetFormatter, load_formatter

FORMATTER_BY_NAME = {
    "json": JsonFormatter,
    "jsonl": JsonFormatter,
    "csv": CsvFormatter,
    "parquet": ParquetFormatter,
}
ARROW_SUFFIXES = {"arrow", "feather"}


def load_source_dataset(source: DataSourceConfig, *, num_proc: int = 1) -> NestedDataset:
    source_format = (source.format or "").lower()
    if source_format in ARROW_SUFFIXES or source.path.lower().endswith((".arrow", ".feather")):
        return _load_arrow_dataset(source)

    formatter_cls = FORMATTER_BY_NAME.get(source_format) if source_format else None
    if formatter_cls is not None:
        formatter = formatter_cls(source.path, text_keys=None, add_suffix=True, **(source.load_options or {}))
    else:
        formatter = load_formatter(source.path, text_keys=None, add_suffix=True, **(source.load_options or {}))
    return NestedDataset(formatter.load_dataset(num_proc=num_proc))


def normalize_source_dataset(source: DataSourceConfig, *, num_proc: int = 1) -> NestedDataset:
    dataset = load_source_dataset(source, num_proc=num_proc)
    adapter = AgentSampleAdapter(source.adapter)
    normalized = dataset.map(
        lambda sample, idx: adapter.adapt_record(sample, row_idx=idx, source=source),
        with_indices=True,
        num_proc=num_proc,
        desc=f"Normalizing {source.path}",
    )
    return NestedDataset(normalized)


def load_and_normalize_sources(sources: Sequence[DataSourceConfig], *, num_proc: int = 1) -> NestedDataset:
    datasets: List[NestedDataset] = [normalize_source_dataset(source, num_proc=num_proc) for source in sources]
    if len(datasets) == 1:
        return datasets[0]
    return NestedDataset(concatenate_datasets(datasets))


def _load_arrow_dataset(source: DataSourceConfig) -> NestedDataset:
    if source.path.lower().endswith(".feather") or (source.format and source.format.lower() == "feather"):
        table = feather.read_table(source.path)
    else:
        with pa.memory_map(source.path, "r") as mapped:
            try:
                table = pa.ipc.open_file(mapped).read_all()
            except pa.ArrowInvalid:
                mapped.seek(0)
                table = pa.ipc.open_stream(mapped).read_all()
    return NestedDataset(Dataset.from_dict(table.to_pydict()))
