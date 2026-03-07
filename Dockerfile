FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY data-juicer /app/data-juicer
COPY agent_posttrain_pipeline /app/agent_posttrain_pipeline
COPY monitoring /app/monitoring
COPY scripts /app/scripts
COPY configs /app/configs
COPY examples /app/examples
COPY tests /app/tests
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml
COPY run_agent_posttrain_pipeline.py /app/run_agent_posttrain_pipeline.py
COPY Makefile /app/Makefile

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ./data-juicer \
    && python -m pip install -e .

CMD ["python", "run_agent_posttrain_pipeline.py", "--config", "configs/dev.yaml"]
