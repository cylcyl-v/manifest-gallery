#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Netlify Pages 20,000 文件限制友好版
- 仅收集候选 -> 统一排序（优先最新 ckpt/子目录）-> 截断到 max-items -> 再复制 & 写 manifest
- 默认 max-items = 19000（给 index.html/JS/CSS 留余量）；可改
- 可用 --latest-iters 仅扫描最近 N 个迭代目录（数字名更可靠）
"""

import argparse, sys, json, shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# --------------------- 基础工具 ---------------------

def read_json(path: Path):
    for enc in ("utf-8","utf-8-sig"):
        try:
            with open(path,"r",encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def extract_prompt(obj):
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                v = el.get("value")
                if isinstance(v,str) and v.strip():
                    return v.strip()
        return ""
    if isinstance(obj, dict):
        for k in ["prompt","instruction","input","question","caption","text"]:
            v = obj.get(k)
            if isinstance(v,str) and v.strip():
                return v.strip()
        for k in ["data","meta","inputs","message","messages","payload"]:
            v = obj.get(k)
            if isinstance(v,(list,dict)):
                p = extract_prompt(v)
                if p:
                    return p
    return ""

def copy_to_site(src: Path, dst_dir: Path, site_root: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if (not dst.exists()) or (src.stat().st_mtime > dst.stat().st_mtime) or (src.stat().st_size != dst.stat().st_size):
        shutil.copy2(src, dst)
    # 相对站点根目录（site/）返回，如 "images/geneval/iter/sub/xxx.jpg"
    return str(dst.relative_to(site_root)).replace("\\", "/")

def first_jpg(folder: Path) -> Optional[Path]:
    files = sorted(folder.glob("*.jpg"))
    return files[0] if files else None

def parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

def iter_dirs_numeric_sorted(root: Path, reverse=True) -> List[Path]:
    """优先按数字名排序；否则用 mtime 回退。"""
    dirs = [p for p in root.iterdir() if p.is_dir()]
    with_num = [(p, parse_int(p.name)) for p in dirs]
    numeric = [p for p,n in with_num if n is not None]
    nonnum = [p for p,n in with_num if n is None]
    numeric.sort(key=lambda p: int(p.name), reverse=reverse)
    nonnum.sort(key=lambda p: p.stat().st_mtime, reverse=reverse)
    return numeric + nonnum

def child_dirs_numeric_sorted(root: Path, reverse=True) -> List[Path]:
    return iter_dirs_numeric_sorted(root, reverse=reverse)

def count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())

def infer_model(p: str, fallback: str):
    parts = Path(p).parts
    if "UnifyModelEval" in parts:
        i = parts.index("UnifyModelEval")
        if i+1 < len(parts):
            return parts[i+1]
    return fallback or "unknown-model"

# --------------------- 候选项收集（不复制） ---------------------

def make_candidate(
    jpg: Path,
    dst_dir: Path,
    benchmark: str,
    iter_dir: Path,
    sub_dir: Path,
    model: str,
    prompt: str
) -> Dict[str, Any]:
    # 排序优先级键
    iter_num = parse_int(iter_dir.name)
    sub_num = parse_int(sub_dir.name)
    mtime = jpg.stat().st_mtime
    # ID 保持可追溯（兼容你原结构）
    item_id = f"{benchmark}-{iter_dir.name}-{sub_dir.name}-{jpg.stem}"
    return {
        "src": jpg,
        "dst_dir": dst_dir,
        "benchmark": benchmark,
        "iter_dir": iter_dir.name,
        "sub_dir": sub_dir.name,
        "model": model,
        "prompt": prompt,
        "mtime": mtime,
        "iter_num": iter_num if iter_num is not None else -1,
        "sub_num": sub_num if sub_num is not None else -1,
        "id": item_id,
    }

def crawl_geneval_candidates(root: Path, images_out: Path, model: str, latest_iters: Optional[int]) -> List[Dict[str,Any]]:
    cands = []
    # 迭代目录优先最新
    iters = iter_dirs_numeric_sorted(root, reverse=True)
    if latest_iters is not None and latest_iters > 0:
        iters = iters[:latest_iters]
    for iter_dir in iters:
        gen_dir = iter_dir / "generation"
        if not gen_dir.exists(): 
            continue
        for count_dir in child_dirs_numeric_sorted(gen_dir, reverse=True):
            samples = count_dir / "samples"
            if not samples.exists():
                continue
            jpg = first_jpg(samples)
            if not jpg:
                continue
            js = jpg.with_suffix(".json")
            meta = read_json(js) if js.exists() else {}
            prompt = extract_prompt(meta)
            dst_dir = images_out / "geneval" / iter_dir.name / count_dir.name
            cands.append(make_candidate(jpg, dst_dir, "geneval", iter_dir, count_dir, model, prompt))
    return cands

def crawl_dpg_candidates(root: Path, images_out: Path, model: str, latest_iters: Optional[int]) -> List[Dict[str,Any]]:
    cands = []
    iters = iter_dirs_numeric_sorted(root, reverse=True)
    if latest_iters is not None and latest_iters > 0:
        iters = iters[:latest_iters]
    for iter_dir in iters:
        gen_dir = iter_dir / "generation"
        if not gen_dir.exists():
            continue
        for name_dir in child_dirs_numeric_sorted(gen_dir, reverse=True):
            jpg = first_jpg(name_dir)
            if not jpg:
                continue
            js = jpg.with_suffix(".json")
            meta = read_json(js) if js.exists() else {}
            prompt = extract_prompt(meta)
            dst_dir = images_out / "dpg" / iter_dir.name / name_dir.name
            cands.append(make_candidate(jpg, dst_dir, "dpg", iter_dir, name_dir, model, prompt))
    return cands

# --------------------- 主流程 ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", required=True, help="Netlify 站点根目录（需包含 index.html）")
    ap.add_argument("--geneval-root")
    ap.add_argument("--dpg-root")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-items", type=int, default=19000, help="最多拷贝的样本条数（约等于新图片文件数）")
    ap.add_argument("--latest-iters", type=int, default=None, help="只扫描最近 N 个迭代目录（按数字名排序）。不设则扫描全部。")
    ap.add_argument("--file-budget", type=int, default=20000, help="部署总文件预算（用于估算有效上限）")
    ap.add_argument("--dry-run", action="store_true", help="只计算与汇报，不实际复制")
    args = ap.parse_args()

    site = Path(args.site_dir)
    if not (site / "index.html").exists():
        print(f"[ERROR] index.html not found under {site}")
        sys.exit(2)

    images_out = site / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # 收集候选
    candidates: List[Dict[str,Any]] = []

    if args.geneval_root:
        model_g = infer_model(args.geneval_root, args.model)
        candidates += crawl_geneval_candidates(Path(args.geneval_root), images_out, model_g, args.latest_iters)

    if args.dpg_root:
        model_d = infer_model(args.dpg_root, args.model)
        candidates += crawl_dpg_candidates(Path(args.dpg_root), images_out, model_d, args.latest_iters)

    total_candidates = len(candidates)
    if total_candidates == 0:
        print("[OK] No candidates found. Nothing to do.")
        # 也写空 manifest，避免前端报错
        manifest = {
            "meta": {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": []},
            "items": []
        }
        (site / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(0)

    # 统一排序：按 (iter_num desc, sub_num desc, mtime desc, benchmark) —— 即“最新 ckpt 优先”
    candidates.sort(
        key=lambda x: (x["iter_num"], x["sub_num"], x["mtime"], x["benchmark"]),
        reverse=True
    )

    # 估算有效上限：受 max-items 与 file-budget 双重限制（严格不超预算）
    existing_files = count_files(site)
    headroom = args.file_budget - existing_files
    if headroom < 0:
        headroom = 0
    effective_limit = min(args.max_items, headroom)

    selected = candidates[:effective_limit]
    print(f"[INFO] candidates={total_candidates}, existing_files={existing_files}, "
          f"file_budget={args.file_budget}, max_items={args.max_items}, effective_limit={effective_limit}, "
          f"selected={len(selected)}")

    if args.dry_run:
        # 列出前 10 个示例给参考
        for i, c in enumerate(selected[:10]):
            print(f"  #{i+1} {c['benchmark']} iter={c['iter_dir']} sub={c['sub_dir']} src={c['src']}")
        print("[DRY-RUN] Skipped copying files.")
        sys.exit(0)

    # 执行复制并生成 manifest
    items = []
    for c in selected:
        web_rel = copy_to_site(c["src"], c["dst_dir"], site)  # 关键修复：传入 site_root
        # geneval 的第二 tag 用 count:，dpg 用 name:（恢复原有语义）
        second_tag_key = "count" if c["benchmark"] == "geneval" else "name"
        items.append({
            "id": c["id"],
            "benchmark": c["benchmark"],
            "dataset": c["benchmark"],
            "split": "",
            "prompt": c["prompt"],
            "image": web_rel,  # 相对站点根目录的路径，如 images/...
            "reference": "",
            "answer": "",
            "prediction": "",
            "model": c["model"],
            "tags": [f"iter:{c['iter_dir']}", f"{second_tag_key}:{c['sub_dir']}"],
            "extra": {}
        })

    manifest = {
        "meta": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": sorted(list(set(x["benchmark"] for x in items)))
        },
        "items": items
    }
    (site / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {site/'manifest.json'} with {len(items)} items")

if __name__ == "__main__":
    main()
