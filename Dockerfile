FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
ARG INSTALL_OPTIONAL=0
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt \
 && if [ "$INSTALL_OPTIONAL" = "1" ]; then python -m pip install -r requirements-optional.txt; fi
COPY . .
RUN mkdir -p data/uploads data/public data/knowledge_base outputs
EXPOSE 8000 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]