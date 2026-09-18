FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

# Project code the API depends on: FeatureEngineer, LABEL_MAP/
# INVERSE_LABEL_MAP, and the shared config -- same modules the training
# pipeline uses, so feature computation and label mapping stay identical
# between training and serving.
COPY shared/ ./shared/
COPY config/ ./config/
COPY api/ ./api/

RUN uv pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]