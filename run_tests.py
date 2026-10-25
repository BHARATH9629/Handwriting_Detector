"""
run_tests.py
------------
Plain-python smoke tests (no pytest needed) for the hlab package.

Run:   python run_tests.py

Needs the trained models present:
    python train_digits.py
    python evaluate_signature.py
and the CEDAR dataset under datasets/ for the signature/writer tests.
"""

import os
import glob
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PASS, FAIL = 0, 0


def check(name, condition, info=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}  {info}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {info}")


def render_digit(d="3"):
    img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 220)
    except Exception:
        font = ImageFont.load_default()
    draw.text((80, 5), d, fill=255, font=font)
    return img


def main():
    print("\n[preprocess]")
    from hlab.preprocess import preprocess_digit
    arr = preprocess_digit(render_digit("5"))
    check("preprocess_digit shape", arr is not None and arr.shape == (28, 28),
          str(None if arr is None else arr.shape))

    print("\n[digits CNN]")
    try:
        from hlab import digits
        model = digits.load_model()
        d, conf, _ = digits.predict(model, render_digit("3"))
        check("recognise a 3", d == 3, f"got {d} @ {conf*100:.0f}%")
        # multi-digit reading
        img = Image.new("L", (420, 200), 0)
        dr = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arialbd.ttf", 150)
        except Exception:
            font = ImageFont.load_default()
        dr.text((30, 10), "4", fill=255, font=font)
        dr.text((230, 10), "2", fill=255, font=font)
        number, _ = digits.predict_sequence(model, img)
        check("read multi-digit '42'", number == "42", f"got '{number}'")
    except FileNotFoundError:
        check("digit model present", False, "run train_digits.py")

    print("\n[letters CNN]")
    try:
        from hlab import letters
        lmodel = letters.load_model()
        letr, lconf, _ = letters.predict(lmodel, render_digit("A"))
        check("recognise an A", letr == "A", f"got {letr} @ {lconf*100:.0f}%")
    except FileNotFoundError:
        check("letter model present", False, "run train_letters.py")

    print("\n[calculator]")
    try:
        from hlab import calc
        cmodel = calc.load_model()
        img = Image.new("L", (420, 200), 0)
        dr = ImageDraw.Draw(img)
        f = render_digit("7").resize((140, 200)); img.paste(f, (0, 0))
        dr.line([180, 100, 240, 100], fill=255, width=13)        # plus: horizontal
        dr.line([210, 70, 210, 130], fill=255, width=13)         # plus: vertical
        f3 = render_digit("3").resize((140, 200)); img.paste(f3, (280, 0))
        out = calc.recognize_expression(cmodel, img)
        check("compute 7+3 = 10", out["result"] == 10,
              f"read '{out['display']}' = {out['result']}")
    except FileNotFoundError:
        check("calculator model present", False, "run train_calc.py")

    cedar = bool(glob.glob("datasets/full_org/*.png"))
    if not cedar:
        print("\n[signature/writer] SKIPPED — CEDAR dataset not found")
    else:
        print("\n[features]")
        from hlab.features import base_features
        v = base_features(glob.glob("datasets/full_org/original_1_*.png")[0])
        check("base_features finite", np.isfinite(v).all(), f"len={len(v)}")

        print("\n[signature verify]")
        try:
            from hlab import signature
            model = signature.load_model()
            gen = sorted(glob.glob("datasets/full_org/original_3_*.png"))
            forg = sorted(glob.glob("datasets/full_forg/forgeries_3_*.png"))
            refs, gen_q = gen[:10], gen[10]
            p_gen, _ = signature.verify(refs, gen_q, model)
            p_forg, _ = signature.verify(refs, forg[0], model)
            check("genuine scores higher than forgery", p_gen > p_forg,
                  f"genuine={p_gen*100:.0f}%  forgery={p_forg*100:.0f}%")
        except FileNotFoundError:
            check("signature model present", False, "run evaluate_signature.py")

        print("\n[writer id]")
        try:
            from hlab import writer_id
            backup = None
            if os.path.exists(writer_id.WRITERS_PATH):
                backup = writer_id.WRITERS_PATH + ".bak"
                shutil.copy(writer_id.WRITERS_PATH, backup)
            writer_id.forget_all_writers()
            writer_id.enroll_writer("P10", sorted(glob.glob("datasets/full_org/original_10_*.png"))[:8])
            writer_id.enroll_writer("P20", sorted(glob.glob("datasets/full_org/original_20_*.png"))[:8])
            probe = sorted(glob.glob("datasets/full_org/original_10_*.png"))[-1]
            name, score, _ = writer_id.identify_writer(probe)
            check("identify correct writer", name == "P10", f"got {name} @ {score*100:.0f}%")
            writer_id.forget_all_writers()
            if backup:
                shutil.move(backup, writer_id.WRITERS_PATH)
        except FileNotFoundError:
            check("signature model present (for embeddings)", False,
                  "run evaluate_signature.py")

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
