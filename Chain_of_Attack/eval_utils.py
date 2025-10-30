# eval_utils.py (or put in utils.py)
import os, json, csv, tempfile, shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import torch
import clip
from tqdm import tqdm

# Reuse your dataset and VLM registry
from evaluator.data import JSONLImageTextDataset
from evaluator.vlms import REGISTRY as VLM_REGISTRY

def _safe_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")

def _device_and_dtype(device: Optional[str], dtype: Optional[str]) -> Tuple[str, torch.dtype]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dt = torch.float16 if str(dev).startswith("cuda") else torch.float32
    else:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return dev, dt

@torch.no_grad()
def _encode_texts(texts: List[str], clip_model, device: str, batch_size: int = 32) -> torch.Tensor:
    # Handle older clip versions without truncate kwarg
    try:
        toks = clip.tokenize(texts, truncate=True).to(device)
    except TypeError:
        toks = clip.tokenize(texts).to(device)

    feats = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        f = clip_model.encode_text(toks[start:end])
        feats.append(f)
    feats = torch.cat(feats, dim=0)
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats

def _compute_clip_text_text_cosine(results_csv: str,
                                   clip_encoder: str,
                                   device: str,
                                   batch_size: int = 32,
                                   limit: Optional[int] = None) -> float:
    tgt_texts, pred_texts = [], []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "target_text", "collected_text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{results_csv} must have columns: image_id,target_text,collected_text")
        for row in reader:
            tgt_texts.append((row.get("target_text") or "").strip())
            pred_texts.append((row.get("collected_text") or "").strip())
    if limit is not None:
        tgt_texts = tgt_texts[:limit]
        pred_texts = pred_texts[:limit]
    if not tgt_texts:
        raise RuntimeError(f"No rows loaded from results CSV: {results_csv}")

    clip_model, _ = clip.load(clip_encoder, device=device)
    clip_model.eval()
    pred_feats = _encode_texts(pred_texts, clip_model, device=device, batch_size=batch_size)
    tgt_feats  = _encode_texts(tgt_texts,  clip_model, device=device, batch_size=batch_size)
    cos = torch.sum(pred_feats * tgt_feats, dim=1)  # cosine similarity per pair
    return cos.mean().item()

def report_metrics(
    annotations_jsonl: str,                  # path to output/annotations.jsonl
    out_dir: str,                            # attack run directory (where metrics_report.json will be saved)
    vlms: List[str],                         # e.g., ["BLIP", "LLaVA-NeXT", "Qwen2.5-VL-7B"]
    clip_encoders: List[str],                # e.g., ["ViT-B/32", "ViT-L/14"]
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    prompt: str = "Describe the image in detail.",
    max_new_tokens: int = 80,
    limit: Optional[int] = None,
    clip_batch_size: int = 32,
    vlm_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,  # per-VLM extra kwargs if needed
) -> Dict[str, Any]:
    """
    Runs VLM generation on the JSONL pairs and evaluates text-text CLIP cosine for each specified encoder.
    Writes a consolidated metrics_report.json in out_dir and returns it as a dict.
    All intermediate per-VLM results are created in a TemporaryDirectory and cleaned up automatically.
    """
    if not os.path.isfile(annotations_jsonl):
        raise FileNotFoundError(f"annotations.jsonl not found: {annotations_jsonl}")
    os.makedirs(out_dir, exist_ok=True)

    dev, dt = _device_and_dtype(device, dtype)

    # Load dataset once
    ds = JSONLImageTextDataset(jsonl_path=annotations_jsonl, limit=limit)
    if len(ds) == 0:
        raise RuntimeError("No records found in annotations JSONL.")

    report: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": len(ds),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "device": dev,
        "dtype": str(dt).replace("torch.", ""),
        "vlms": {},
    }

    with tempfile.TemporaryDirectory(prefix="eval_tmp_") as tmp_root:
        # For each VLM, run generation and evaluate with each CLIP encoder
        for vlm_name in vlms:
            if vlm_name not in VLM_REGISTRY:
                raise KeyError(f"VLM '{vlm_name}' not registered. Available: {list(VLM_REGISTRY.keys())}")
            vlm_ctor = VLM_REGISTRY[vlm_name]
            per_vlm_kwargs = (vlm_kwargs or {}).get(vlm_name, {})
            # Instantiate model
            print(f"[eval] Loading VLM: {vlm_name}")
            vlm = vlm_ctor(device=dev, dtype=str(dt).replace("torch.", ""), **per_vlm_kwargs)

            # Prepare per-VLM temp folder
            vlm_tag = _safe_name(vlm_name)
            vlm_tmp = os.path.join(tmp_root, f"{vlm_tag}")
            os.makedirs(vlm_tmp, exist_ok=True)
            results_csv = os.path.join(vlm_tmp, f"results_{vlm_tag}.csv")

            # Run generation loop (same columns as your eval script)
            print(f"[eval] Generating with {vlm_name} on {len(ds)} samples...")
            with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["image_id", "target_text", "collected_text"])
                for i in tqdm(range(len(ds)), desc=f"{vlm_name}"):
                    item = ds[i]
                    image_pil = item["image_pil"]
                    image_id = item["image_id"]
                    tgt = item["target_text"] or ""
                    out = vlm.generate(image_pil, prompt=prompt, max_new_tokens=max_new_tokens)
                    pred_txt = (out.text or "").strip()
                    writer.writerow([image_id, tgt, pred_txt])

            # For this results CSV, evaluate with each CLIP encoder
            clip_scores = {}
            for enc in clip_encoders:
                print(f"[eval] CLIP text-text cosine with encoder: {enc}")
                score = _compute_clip_text_text_cosine(
                    results_csv=results_csv,
                    clip_encoder=enc,
                    device=dev,
                    batch_size=clip_batch_size,
                    limit=limit,
                )
                clip_scores[enc] = score

            report["vlms"][vlm_name] = {
                "results_csv_tmp": results_csv,  # will be removed; kept for traceability in-memory
                "clip_text_cosine": clip_scores,
            }

        # All tmp files/dirs will be removed here on exit

    # Save consolidated report to out_dir
    report_path = os.path.join(out_dir, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
    print(f"[eval] Saved consolidated metrics to: {report_path}")

    return report


def _collect_encoders(report):
    encs = []
    seen = set()
    for _, v in (report.get("vlms") or {}).items():
        for enc in (v.get("clip_text_cosine") or {}):
            if enc not in seen:
                seen.add(enc)
                encs.append(enc)
    return encs

def print_metrics_report(report: dict,
                         enc_order: list = None,
                         sort_by: str = "avg",   # "avg" or an encoder name to sort by that column
                         reverse: bool = True,
                         use_rich: bool = True,
                         save_path: str = None):
    """
    Pretty-print the consolidated metrics report.

    - enc_order: if provided, use this encoder column order; else infer from the report.
    - sort_by: "avg" or a specific encoder name; rows are sorted by this column.
    - reverse: True for descending.
    - use_rich: try rich table if installed; fallback to ASCII otherwise.
    """
    vlms = report.get("vlms", {})
    if not vlms:
        print("[eval] No VLM entries found in report.")
        return

    encs = enc_order or _collect_encoders(report)
    # Build rows: [VLM, enc1, enc2, ..., Avg]
    rows = []
    for vlm_name, data in vlms.items():
        scores = data.get("clip_text_cosine", {}) or {}
        row_scores = [scores.get(e) for e in encs]
        # average over available encs
        vals = [v for v in row_scores if isinstance(v, (int, float))]
        avg = sum(vals) / len(vals) if vals else float("nan")
        rows.append((vlm_name, row_scores, avg))

    # Choose sort key
    if sort_by == "avg":
        rows.sort(key=lambda r: (-(r[2] if r[2] == r[2] else -1e9)), reverse=False)  # NaN-safe
        if reverse:
            rows.reverse()
    else:
        # sort by specific encoder column if present
        try:
            idx = encs.index(sort_by)
            rows.sort(key=lambda r: (-(r[1][idx] if isinstance(r[1][idx], (int, float)) else -1e9)), reverse=False)
            if reverse:
                rows.reverse()
        except ValueError:
            pass  # leave unsorted if encoder not found

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "-"  # NaN->"-"

    header = ["VLM"] + encs + ["Avg"]

    # # Try rich first
    # if use_rich:
    #     try:
    #         from rich.console import Console
    #         from rich.table import Table
    #         table = Table(show_lines=False, header_style="bold")
    #         for h in header:
    #             table.add_column(h, justify="center")
    #         for name, s_list, avg in rows:
    #             table.add_row(name, *[_fmt(v) for v in s_list], _fmt(avg))
    #         Console().print("\n[bold]Evaluation Summary[/bold]")
    #         Console().print(f"Samples: {report.get('num_samples')}  Device: {report.get('device')}  DType: {report.get('dtype')}")
    #         Console().print(table)
    #         return
    #     except Exception:
    #         pass
    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(show_lines=False, header_style="bold")
            for h in header:
                table.add_column(h, justify="center")
            for name, s_list, avg in rows:
                table.add_row(name, *[_fmt(v) for v in s_list], _fmt(avg))

            console = Console()
            console.print("\n[bold]Evaluation Summary[/bold]")
            console.print(f"Samples: {report.get('num_samples')}  Device: {report.get('device')}  DType: {report.get('dtype')}")
            console.print(table)

            # ---- NEW: Write rich table text output to file if save_path is provided ----
            if save_path:
                # Capture console output into a string
                from rich.console import Console
                from io import StringIO

                buffer = StringIO()
                temp_console = Console(file=buffer, width=120)
                temp_console.print("\n[bold]Evaluation Summary[/bold]")
                temp_console.print(f"Samples: {report.get('num_samples')}  Device: {report.get('device')}  DType: {report.get('dtype')}")
                temp_console.print(table)

                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(buffer.getvalue())
                print(f"\n[Saved metrics table to: {save_path}]")
            # -------------------------------------------------------------------------

            return
        except Exception:
            pass

    # Fallback: ASCII table
    col_widths = [len(h) for h in header]
    data_rows = []
    for name, s_list, avg in rows:
        cells = [name] + [_fmt(v) for v in s_list] + [_fmt(avg)]
        data_rows.append(cells)
        col_widths = [max(w, len(c)) for w, c in zip(col_widths, cells)]

    def line(parts, widths, sep=" | "):
        return sep.join(p.ljust(w) for p, w in zip(parts, widths))

    sep_line = "-+-".join("-" * w for w in col_widths)
    print("\nEvaluation Summary")
    print(f"Samples: {report.get('num_samples')}  Device: {report.get('device')}  DType: {report.get('dtype')}")
    print(line(header, col_widths))
    print(sep_line)
    for cells in data_rows:
        print(line(cells, col_widths))