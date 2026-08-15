FROM python:3.11-slim

WORKDIR /code
COPY pyproject.toml README.md ./
COPY footnote ./footnote
COPY app ./app
COPY prompts ./prompts
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[api]" && mkdir -p data && chmod -R 777 data

# corpus is fetched from the Publications Office on first boot and cached in data/
# PORT is injected by the host (Render); 7860 is the local default
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
