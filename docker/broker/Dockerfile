# Broker-only image for the PirateBot prop mesh.
# Serves WebSocket mesh, UDP discovery, control center, and display static files.

FROM python:3.12-slim

# Use local APT cache proxy if available (offline-friendly).
RUN echo 'Acquire::http::Proxy "http://192.168.0.2:3142";' > /etc/apt/apt.conf.d/00proxy || true
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use local devpi mirror for offline-friendly installs.
RUN pip install --no-cache-dir \
    --index-url http://192.168.0.2:4040/root/pypi/+simple/ \
    --extra-index-url https://pypi.org/simple \
    --trusted-host 192.168.0.2 \
    aiohttp>=3.9.0 \
    aiomqtt>=2.0.0 \
    cbor2>=5.6.0 \
    pyyaml>=6.0

COPY props/broker /app/props/broker
COPY props/lib /app/props/lib
COPY props/control_center/dist /app/props/control_center/dist
COPY props/displays/garage_ship/dist /app/props/displays/garage_ship/dist
COPY props/displays/pumpkins/dist /app/props/displays/pumpkins/dist

ENV PYTHONPATH=/app

EXPOSE 9000 9001 9002/udp

CMD ["python", "-m", "props.broker.mesh_broker"]
