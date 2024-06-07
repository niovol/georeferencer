FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY requirements.txt /tmp
RUN apt update && \
    apt install -y \
    libgl1-mesa-glx && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 8000