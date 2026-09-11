#Construir la imagen del servicio

#La imagen base

FROM python:3.12-slim

#La carpeta donde va a vivir todo dentro del contenedor

WORKDIR /servicio

#Que los mensajes salgan inmediatamente

ENV PYTHONDONTWRITEBYTECODE=1 \ 
    PYTHONUNBUFFERED=1

    
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \ 
    && rm -rf /var/lib/apt/lists/*

#Copiar solo el requirements

COPY requirements.txt .

#Instalamos las librerías

RUN pip install --no-cache-dir -r requirements.txt

#Copiamos todos los archivos
COPY . .

#Documentación de puerto

EXPOSE 8080

#Correr el despliegue

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]