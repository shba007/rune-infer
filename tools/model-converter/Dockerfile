FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 HF_HOME=/work/cache HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    VIRTUAL_ENV=/opt/converter-venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /converter
COPY requirements.lock ./
RUN uv venv --python 3.11 /opt/converter-venv
# Preserve the original merged-index selection with every version pinned.
RUN --mount=type=cache,target=/root/.cache/uv uv pip sync \
    --python /opt/converter-venv/bin/python --index-strategy unsafe-best-match requirements.lock
RUN uv pip check --python /opt/converter-venv/bin/python
COPY convert_depth.py reference_depth.py ./
ENTRYPOINT ["uv", "run", "--no-project", "--python", "/opt/converter-venv/bin/python", "/converter/convert_depth.py"]
