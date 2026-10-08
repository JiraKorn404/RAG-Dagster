import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
# The experiment a PDF put into data/raw is ingested with (read by the sensor at every tick).
INGEST_CONFIG = Path(os.environ.get("INGEST_CONFIG", "config/ingest.yaml"))


def artifacts_dir(experiment: str, stage: str) -> Path:
    """Per-stage outputs: data/artifacts/<experiment>/<stage>/. Created on demand."""
    path = DATA_DIR / "artifacts" / experiment / stage
    path.mkdir(parents=True, exist_ok=True)
    return path
