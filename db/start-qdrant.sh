#!/bin/bash

# Create directories if they don't exist
mkdir -p data

docker run -d \
  --name bioasq-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/data:/qdrant/storage \
  -e QDRANT_ALLOW_RECOVERY=true \
  --restart unless-stopped \
  qdrant/qdrant
