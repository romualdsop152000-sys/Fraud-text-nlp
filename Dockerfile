FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Toutes les dépendances sauf torch
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    nltk \
    transformers \
    joblib \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    elasticsearch

# Torch CPU séparément
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]