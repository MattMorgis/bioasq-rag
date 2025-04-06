# Qdrant Vector Database for BioASQ RAG

This directory contains the setup for Qdrant, the vector database used by the BioASQ RAG system.

## Local Setup

### Prerequisites

- Docker installed
- Python 3.8+ with pip

### Running Qdrant Locally

1. Start the Qdrant server:

```bash
cd db
chmod +x run-qdrant.sh
./run-qdrant.sh
```

2. Verify that Qdrant is running:

```bash
# Check container status
docker ps | grep bioasq-qdrant

# Check logs if needed
docker logs bioasq-qdrant

# Or use the provided Python utility
python qdrant_client.py
```

3. Access the Qdrant dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Stopping Qdrant

```bash
./stop-qdrant.sh
# or manually
docker stop bioasq-qdrant
docker rm bioasq-qdrant
```

## Python Client

The `qdrant_client.py` file provides a simple utility for connecting to Qdrant from Python. 
To use it in other modules:

```python
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the client
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from db.qdrant_client import get_qdrant_client

# Create a client
client = get_qdrant_client()

# Use the client to interact with Qdrant
# Example: create a collection
# client.create_collection(...)
```

## Configuration

The `config.yaml` file contains the Qdrant configuration. You can modify settings such as:

- Storage paths
- Optimization parameters 
- HNSW search configuration
- Port settings

After changing the configuration, restart the container:

```bash
docker-compose down
docker-compose up -d
```

## Digital Ocean Deployment

See `digital-ocean-setup.md` for detailed deployment options, including:

1. Simple Docker run on a droplet
2. Systemd service for better process management
3. Digital Ocean App Platform

Remember to:
- Set up appropriate firewall rules
- Consider using a volume or block storage for persistence
- Configure necessary credentials for production use

## Common Tasks

### Backup Data

To backup Qdrant data:

```bash
# Create a snapshot
curl -X POST 'http://localhost:6333/snapshots'

# List available snapshots
curl 'http://localhost:6333/snapshots'

# Download a snapshot (replace SNAPSHOT_NAME)
curl -X GET 'http://localhost:6333/snapshots/SNAPSHOT_NAME' -o snapshot.zip
```

### Monitoring

The Qdrant dashboard provides basic monitoring. For more advanced needs, consider:

- Setting up Prometheus and Grafana
- Using the health endpoint (`/healthz`) for regular checks
