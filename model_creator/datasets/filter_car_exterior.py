#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from transformers import pipeline, CLIPProcessor, CLIPModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def find_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def load_models(device: Optional[str] = None):
    """
    Load DETR for object detection and CLIP for exterior vs interior scoring.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Object detection: DETR (labels include "car")
    det_pipe = pipeline(
        task="object-detection",
        model="facebook/detr-resnet-50",
        device=0 if device == "cuda" else -1,
    )

    # CLIP for semantic disambiguation (exterior vs interior)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = clip_model.to(device).eval()

    return det_pipe, clip_model, clip_processor, device

def detect_car(det_pipe, img: Image.Image, min_conf: float) -> Tuple[bool, float, float]:
    """
    Returns (has_car, best_confidence_for_car, best_bbox_area_ratio)
    """
    preds = det_pipe(img)
    width, height = img.size
    img_area = float(width * height)

    best_conf = 0.0
    best_area_ratio = 0.0
    has_car = False

    for det in preds:
        label = det.get("label", "").lower()
        score = float(det.get("score", 0.0))
        box = det.get("box", {})
        if not all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
            continue
        w = max(0.0, float(box["xmax"] - box["xmin"]))
        h = max(0.0, float(box["ymax"] - box["ymin"]))
        area_ratio = (w * h) / img_area if img_area > 0 else 0.0

        if label == "car" and score >= min_conf:
            has_car = True
            if score > best_conf:
                best_conf = score
            if area_ratio > best_area_ratio:
                best_area_ratio = area_ratio

    return has_car, best_conf, best_area_ratio

def clip_exterior_score(clip_model, clip_processor, device: str, img: Image.Image) -> Tuple[float, float]:
    """
    Returns (exterior_score, interior_score) as softmax probabilities over two prompts.
    """
    prompts = [
        "a photo of the EXTERIOR of a car",
        "a photo of the INTERIOR of a car (dashboard, seats, steering wheel)",
        "a close-up of a car part (wheel, headlight, emblem)",
        "a toy car or miniature model",
        "a car in a video game or illustration",
    ]
    # We'll take the first as "exterior", second as "interior", others as distractors.
    with torch.no_grad():
        inputs = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        # logits_per_image: [1, num_texts]
        logits = outputs.logits_per_image.squeeze(0)  # shape: [num_texts]
        probs = logits.softmax(dim=-1).detach().cpu().tolist()

    exterior = float(probs[0])
    interior = float(probs[1])
    return exterior, interior

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Filtrer des images contenant une voiture vue de l'extérieur.")
    parser.add_argument("input_dir", type=str, help="Dossier source contenant les images.")
    parser.add_argument("output_dir", type=str, help="Dossier de sortie pour copier les images retenues.")
    parser.add_argument("--recursive", action="store_true", help="Parcourir récursivement les sous-dossiers.")
    parser.add_argument("--min_conf", type=float, default=0.7, help="Score minimum DETR pour la classe 'car'. (defaut: 0.7)")
    parser.add_argument("--min_area", type=float, default=0.12, help="Surface min. du meilleur cadre 'car' (ratio 0-1). (defaut: 0.12)")
    parser.add_argument("--exterior_margin", type=float, default=0.10, help="Marge minimale (exterior - interior) pour valider extérieur. (defaut: 0.10)")
    parser.add_argument("--report_csv", type=str, default="car_filter_report.csv", help="Chemin du CSV de rapport.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERREUR] Dossier d'entrée introuvable: {in_dir}", file=sys.stderr)
        sys.exit(1)

    det_pipe, clip_model, clip_processor, device = load_models()

    images = find_images(in_dir, args.recursive)
    if not images:
        print("[INFO] Aucune image trouvée.")
        sys.exit(0)

    # Prepare CSV report
    import csv
    report_path = Path(args.report_csv).expanduser().resolve()
    f_report = open(report_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_report)
    writer.writerow(["image_path", "has_car", "car_best_conf", "car_best_area_ratio", "exterior_score", "interior_score", "decision"])

    kept = 0
    errors = 0
    for img_path in tqdm(images, desc="Analyse des images"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            errors += 1
            writer.writerow([str(img_path), "error", "", "", "", "", f"open_error:{e}"])
            continue

        has_car, best_conf, best_area_ratio = detect_car(det_pipe, img, args.min_conf)

        if not has_car or best_area_ratio < args.min_area:
            writer.writerow([str(img_path), has_car, f"{best_conf:.4f}", f"{best_area_ratio:.4f}", "", "", "reject:no_car_or_small_bbox"])
            continue

        exterior, interior = clip_exterior_score(clip_model, clip_processor, device, img)
        decision = "keep" if (exterior - interior) >= args.exterior_margin else "reject:likely_interior_or_ambiguous"

        writer.writerow([
            str(img_path),
            has_car,
            f"{best_conf:.4f}",
            f"{best_area_ratio:.4f}",
            f"{exterior:.4f}",
            f"{interior:.4f}",
            decision
        ])

        if decision == "keep":
            # Maintain relative structure if recursive
            if args.recursive:
                rel = img_path.relative_to(in_dir)
                target_dir = out_dir / rel.parent
                ensure_dir(target_dir)
                shutil.copy2(img_path, target_dir / img_path.name)
            else:
                shutil.copy2(img_path, out_dir / img_path.name)
            kept += 1

    f_report.close()

    print(f"[OK] Terminé. {kept} image(s) copiée(s) vers {out_dir}. Rapport: {report_path}. Erreurs de lecture: {errors}")

if __name__ == "__main__":
    main()
