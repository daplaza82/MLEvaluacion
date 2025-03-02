FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear las carpetas necesarias
RUN mkdir -p data/raw data/processed models reports/figures

# Puerto para MLflow UI
EXPOSE 5000

# Comando por defecto para ejecutar MLflow server
CMD ["mlflow", "ui", "--host", "0.0.0.0"]