"""Amostra reprodutível de títulos únicos de um data/raw/*.ibyte.json.

Uso:
    python scripts/2.sample_titles.py data/raw/cellphone.ibyte.json --n 300 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def extract_titles(raw_path: Path) -> list[str]:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    titles = []
    seen = set()
    for item in data:
        for microdata in item.get("microdata", []):
            if microdata.get("@type") == "Product":
                name = microdata.get("name")
                if name and name not in seen:
                    seen.add(name)
                    titles.append(name)
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    titles = extract_titles(args.raw_path)
    print(f"{len(titles)} unique titles found in {args.raw_path}")

    random.seed(args.seed)
    sample = random.sample(titles, args.n)
    for title in sample:
        print(title)


if __name__ == "__main__":
    main()
