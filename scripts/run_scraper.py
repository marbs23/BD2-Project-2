"""
Sprint 1 — Ingesta de imágenes BBC News
Uso:
    python scripts/run_scraper.py --limit 200   # prueba rápida
    python scripts/run_scraper.py               # dataset completo
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.image.scraper import run

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default=str(ROOT / "data/raw/bbc_news.csv"))
    parser.add_argument("--images",  default=str(ROOT / "data/raw/images"))
    parser.add_argument("--meta",    default=str(ROOT / "data/raw/metadata.json"))
    parser.add_argument("--limit",   type=int, default=None, help="Máx artículos (None = todos)")
    parser.add_argument("--delay",   type=float, default=1.0, help="Segundos entre requests")
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        images_dir=args.images,
        meta_path=args.meta,
        limit=args.limit,
        delay=args.delay,
        only_news=True,
    )
