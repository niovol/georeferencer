FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY requirements.txt /tmp
RUN apt update && \
    apt install -y \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    python3-gdal \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8000