
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm
import numpy as np

def export_cars196(output_dir: Path, img_format: str = "jpg", seed: int = 42, quality: int = 95):
    """
    Télécharge Cars196 via TFDS, regroupe train+test, mélange et exporte
    les images en 0.jpg, 1.jpg, ... dans output_dir.
    """
    img_format = img_format.lower()
    if img_format not in {"jpg", "jpeg", "png"}:
        raise ValueError("img_format doit être 'jpg' ou 'png'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Téléchargement/chargement du dataset (images + labels)
    # as_supervised=True -> (image, label)
    ds_train = tfds.load("cars196", split="train", as_supervised=True)
    ds_test  = tfds.load("cars196", split="test",  as_supervised=True)

    # Concatène les deux splits
    ds_all = ds_train.concatenate(ds_test)

    # Convertit en liste de tableaux numpy pour mélanger facilement
    images = []
    for image, _ in tfds.as_numpy(ds_all):
        # image est un np.ndarray (H, W, 3) uint8
        images.append(image)

    # Mélange
    rnd = random.Random(seed)
    rnd.shuffle(images)

    # Export
    count = 0
    ext = ".jpg" if img_format == "jpg" else ".png"
    for arr in tqdm(images, desc="Export des images"):
        # Garantit le type et l'intervalle
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        im = Image.fromarray(arr)
        out_path = output_dir / f"{count}{ext}"

        if img_format == "jpg":
            # Convertit en RGB si besoin pour JPEG
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_path, format="JPEG", quality=quality, optimize=True)
        else:
            im.save(out_path, format="PNG", optimize=True)

        count += 1

    print(f"✅ Terminé : {count} images exportées dans {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Télécharger Cars196 et exporter toutes les images en fichiers numérotés.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Dossier de sortie (sera créé s'il n'existe pas).")
    parser.add_argument("-f", "--format", type=str, default="jpg", choices=["jpg", "png", "jpeg"],
                        help="Format d'export : jpg ou png (par défaut: jpg).")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour le shuffle (par défaut: 42).")
    parser.add_argument("--quality", type=int, default=95,
                        help="Qualité JPEG (1-100, par défaut: 95). Ignoré pour PNG.")
    args = parser.parse_args()

    export_cars196(
        output_dir=Path(args.output),
        img_format=args.format,
        seed=args.seed,
        quality=args.quality
    )

if __name__ == "__main__":
    main()
