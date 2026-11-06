# API 與 worker 共用同一個映像檔，靠啟動指令區分角色（見 docker-compose.yml）
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/srv/.venv/bin:$PATH"

# 先只裝相依套件：程式碼變動時不必重裝
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
