#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unified manifest.json for the static viewer from Geneval / DPG outputs.

Usage (examples):
  # 1) Generic CSV/JSON adapter with obvious columns
  python scripts/build_manifest.py \
    --out manifest.json \
    --geneval /path/to/geneval_results.csv \
    --geneval-image-root /path/to/geneval/images \
    --dpg /path/to/dpg_results.csv \
    --dpg-image-root /path/to/dpg/gens

  # 2) JSONL inputs:
  python scripts/build_manifest.py \
    --out manifest.json \
    --geneval /path/to/geneval_results.jsonl \
    --dpg /path/to/dpg_results.jsonl

Expectations:
  We try to auto-detect columns:
    id, prompt, image / img / gen_path, model, dataset, split, score, reference, answer, prediction, tags
  For relative image paths, we prefix with the given *-image-root if provided.
  Unknown columns are packed into `extra`.
"""
import argparse, csv, json, os, sys, pathlib
from typing import List, Dict, Any, Optional

def read_table(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k,v in row.items()}
                rows.append(row)
    elif p.suffix.lower() in (".jsonl", ".jl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rows.append(json.loads(line))
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list): rows = obj
            elif isinstance(obj, dict) and "items" in obj: rows = obj["items"]
            else: raise ValueError("Unrecognized JSON structure for table: expected list or {items:[]}")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return rows

def norm_item(src: Dict[str, Any], image_root: Optional[str], benchmark_hint: str) -> Dict[str, Any]:
    # heuristic field mapping
    def pick(*keys, default=""):
        for k in keys:
            if k in src and src[k] not in (None, ""): return src[k]
        return default
    # image path
    img = pick("image","img","gen_path","pred_path","output_path","image_path")
    if img and image_root and not (str(img).startswith("http") or str(img).startswith("/") or str(img).startswith("./")):
        img = str(pathlib.Path(image_root) / str(img))
    ref = pick("reference","ref_image","gt_image","target_path","ref_path")
    if ref and image_root and not (str(ref).startswith("http") or str(ref).startswith("/") or str(ref).startswith("./")):
        ref = str(pathlib.Path(image_root) / str(ref))

    score = pick("score","metric","overall","fid","clip","acc","accuracy", default=None)
    try:
        score = float(score) if score not in ("", None) else None
    except Exception:
        score = None

    known = {"id","prompt","image","img","gen_path","pred_path","output_path","image_path",
             "model","dataset","split","score","reference","ref_image","gt_image",
             "answer","prediction","tags"}
    extra = {k:v for k,v in src.items() if k not in known}

    item = {
        "id": pick("id","sample_id","qid","uid","name", default=""),
        "benchmark": pick("benchmark", default=benchmark_hint),
        "dataset": pick("dataset","task","subset","category","bench", default=""),
        "split": pick("split","phase","partition", default=""),
        "prompt": pick("prompt","instruction","input","question","caption","text", default=""),
        "image": img,
        "reference": ref,
        "answer": pick("answer","gt","ground_truth"),
        "prediction": pick("prediction","pred","output"),
        "score": score,
        "model": pick("model","model_name","ckpt","run"),
        "tags": pick("tags", default=[]),
        "extra": extra
    }
    return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output manifest.json")
    ap.add_argument("--geneval", help="Geneval results file (csv/json/jsonl)")
    ap.add_argument("--geneval-image-root", help="Prefix for relative image paths in Geneval results")
    ap.add_argument("--dpg", help="DPG results file (csv/json/jsonl)")
    ap.add_argument("--dpg-image-root", help="Prefix for relative image paths in DPG results")
    args = ap.parse_args()

    items: List[Dict[str, Any]] = []

    if args.geneval:
        for row in read_table(args.geneval):
            items.append(norm_item(row, args.geneval_image_root, "geneval"))
    if args.dpg:
        for row in read_table(args.dpg):
            items.append(norm_item(row, args.dpg_image_root, "dpg"))

    manifest = {
        "meta": {"created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "items": items
    }
    outp = pathlib.Path(args.out)
    outp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {outp} with {len(items)} items")

if __name__ == "__main__":
    main()
