#!/usr/bin/env python3
"""Summarize the five LLM-as-a-Judge metrics from one or more JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    ("Reliability", "rewrite_rel_acc"),
    ("T-Generality", "rephrase_rel_acc"),
    ("M-Generality", "rephrase_image_rel_acc"),
    ("T-Locality", "locality_rel_acc"),
    ("M-Locality", "multimodal_locality_rel_acc"),
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, nargs="+")
    args = parser.parse_args()

    header = ["File", "Rows", *(name for name, _ in METRICS), "Average"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---", "---:"] + ["---:"] * (len(header) - 2)) + "|")
    for path in args.jsonl:
        rows = read_rows(path)
        values = []
        for _, key in METRICS:
            metric = [
                float(row["post"][key])
                for row in rows
                if isinstance(row.get("post", {}).get(key), (int, float))
            ]
            values.append(sum(metric) / len(metric) if metric else float("nan"))
        average = sum(values) / len(values)
        fields = [
            path.name,
            str(len(rows)),
            *(f"{100 * value:.2f}" for value in values),
            f"{100 * average:.2f}",
        ]
        print("| " + " | ".join(fields) + " |")


if __name__ == "__main__":
    main()
