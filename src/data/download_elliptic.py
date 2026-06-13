"""Download the Elliptic Bitcoin dataset (Phase 1.1).

The Elliptic dataset is a real, labelled Bitcoin transaction graph:
~203k transaction nodes, ~234k directed payment edges, 166 features/node, and
licit / illicit / unknown labels across 49 time steps.

Source (free, no auth): the PyTorch Geometric data mirror at
``https://data.pyg.org/datasets/elliptic`` — the same three CSVs that Kaggle
hosts under ``ellipticco/elliptic-data-set``, served as zips. A Kaggle-API
fallback is documented in ``data/raw/elliptic/README`` for offline mirrors.

Files downloaded into ``data/raw/elliptic/``:
    elliptic_txs_features.csv   node id, time step, 165 features
    elliptic_txs_classes.csv    node id, class (1=illicit, 2=licit, unknown)
    elliptic_txs_edgelist.csv   txId1 -> txId2 directed edges

Usage:
    python -m src.data.download_elliptic
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("download_elliptic")

# Repo root = three levels up from this file (src/data/download_elliptic.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "elliptic"

PYG_BASE_URL = "https://data.pyg.org/datasets/elliptic"
CSV_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` with a simple progress log."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:  # noqa: S310 (trusted, hard-coded https host)
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(dest, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                read += len(block)
                if total:
                    logger.info(
                        "  %s: %5.1f%% (%d/%d bytes)",
                        dest.name,
                        100 * read / total,
                        read,
                        total,
                    )
    logger.info("Saved %s (%d bytes)", dest.name, dest.stat().st_size)


def download() -> Path:
    """Download + extract all three Elliptic CSVs. Returns the raw dir."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for csv_name in CSV_FILES:
        csv_path = RAW_DIR / csv_name
        if csv_path.exists():
            logger.info("Already present, skipping: %s", csv_name)
            continue

        zip_url = f"{PYG_BASE_URL}/{csv_name}.zip"
        zip_path = RAW_DIR / f"{csv_name}.zip"
        logger.info("Downloading %s", zip_url)
        _download(zip_url, zip_path)

        logger.info("Extracting %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()  # drop the zip, keep the CSV

    return RAW_DIR


def verify(raw_dir: Path) -> dict[str, int]:
    """Light sanity check on row counts (no pandas needed)."""
    counts: dict[str, int] = {}
    for csv_name in CSV_FILES:
        path = raw_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")
        # rows excluding the header line
        with open(path, "r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        counts[csv_name] = n
    return counts


def main() -> int:
    logger.info("Elliptic Bitcoin dataset -> %s", RAW_DIR)
    try:
        raw_dir = download()
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)
        logger.error(
            "Fallback: install kaggle, place kaggle.json, then run\n"
            "  kaggle datasets download -d ellipticco/elliptic-data-set "
            "-p data/raw/elliptic --unzip"
        )
        return 1

    counts = verify(raw_dir)
    logger.info("Row counts (excluding headers):")
    for name, n in counts.items():
        logger.info("  %-30s %8d", name, n)

    nodes = counts["elliptic_txs_classes.csv"]
    edges = counts["elliptic_txs_edgelist.csv"]
    logger.info("Nodes (labelled+unlabelled): %d | Edges: %d", nodes, edges)
    if nodes < 100_000:
        logger.warning("Node count below the 100k Phase 1.1 target.")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
