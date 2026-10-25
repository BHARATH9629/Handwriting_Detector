"""
hlab.report
-----------
Visual helpers: turn results into pictures people can actually understand.

  * signature_comparison_image — genuine vs questioned vs a red difference map
  * explanation_text           — a plain-English sentence about the verdict
  * save_signature_report      — write a PNG + TXT report to disk
  * digit_view_strip           — show the cleaned image the digit CNN sees
"""

from __future__ import annotations

import os
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .preprocess import normalize_signature, preprocess_digit


def _font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _panel(img_arr, title, w=256, h=128):
    img = (Image.fromarray(img_arr).convert("RGB") if img_arr.ndim == 2
           else Image.fromarray(img_arr))
    img = img.resize((w, h))
    out = Image.new("RGB", (w, h + 22), "white")
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((3, 3), title, fill="black", font=_font(13))
    return out


def signature_comparison_image(genuine_paths, questioned_path):
    """Side-by-side [average genuine | questioned | difference heatmap]."""
    gens = [normalize_signature(p)["gray"] for p in genuine_paths]
    avg = np.mean(np.stack(gens), axis=0).astype(np.uint8)
    q = normalize_signature(questioned_path)["gray"]

    diff = cv2.absdiff(avg, q)
    heat = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    panels = [_panel(avg, "Genuine (average)"),
              _panel(q, "Questioned"),
              _panel(heat, "Differences (red = differ)")]
    gap = 10
    W = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (W, panels[0].height), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def explanation_text(prob, verdict):
    pct = f"{prob*100:.0f}%"
    if verdict.startswith("LIKELY GENUINE"):
        return (f"Match score {pct}. Stroke shapes and slant line up well with "
                f"the genuine references (few large red areas).")
    if verdict.startswith("BORDERLINE"):
        return (f"Match score {pct}. Close to the decision line — some areas "
                f"differ. Best to have a person double-check.")
    return (f"Match score {pct}. Shapes/slant differ noticeably from the "
            f"genuine references (see red areas) — looks like a forgery.")


def save_signature_report(genuine_paths, questioned_path, prob, verdict,
                          out_dir="reports"):
    """Save a PNG (with picture + text) and a TXT summary. Returns the PNG path."""
    os.makedirs(out_dir, exist_ok=True)
    comp = signature_comparison_image(genuine_paths, questioned_path)

    header = 64
    out = Image.new("RGB", (comp.width, comp.height + header), "white")
    out.paste(comp, (0, header))
    d = ImageDraw.Draw(out)
    color = {"LIKELY GENUINE": (26, 127, 55),
             "BORDERLINE — check manually": (154, 103, 0)}.get(verdict, (180, 35, 24))
    d.text((6, 6), f"Signature check: {verdict}", fill=color, font=_font(20))
    d.text((6, 34), explanation_text(prob, verdict), fill=(60, 60, 60), font=_font(12))

    ts = time.strftime("%Y%m%d_%H%M%S")
    png = os.path.join(out_dir, f"signature_report_{ts}.png")
    out.save(png)
    with open(os.path.join(out_dir, f"signature_report_{ts}.txt"), "w",
              encoding="utf-8") as f:
        f.write("Signature check report\n" + "=" * 30 + "\n")
        f.write(f"Verdict: {verdict}\nGenuine-probability: {prob*100:.1f}%\n")
        f.write(f"Genuine references: {len(genuine_paths)}\n"
                f"Questioned file: {questioned_path}\n\n")
        f.write(explanation_text(prob, verdict) + "\n")
    return png


def digit_view_strip(canvas_image, boxes, cell=64):
    """Render each detected digit as the cleaned 28x28 image the CNN receives."""
    full = np.asarray(canvas_image.convert("L"), dtype=np.uint8)
    crops = ([full] if not boxes
             else [full[y:y + h, x:x + w] for (x, y, w, h) in boxes])
    tiles = []
    for c in crops:
        a = preprocess_digit(Image.fromarray(c))
        if a is not None:
            tiles.append((a * 255).astype(np.uint8))
    if not tiles:
        return None
    imgs = [Image.fromarray(t).resize((cell, cell), Image.NEAREST) for t in tiles]
    gap = 6
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("L", (W, cell), 0)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out.convert("RGB")
