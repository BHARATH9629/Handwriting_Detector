"""
web_app.py
----------
Web version of HandwritingLab. Same brains (the `hlab` package), but the UI
runs in a browser instead of a desktop window — so you can use it from your
phone too (on the same Wi-Fi).

Run:   python web_app.py
Then open the printed address (http://localhost:5000) in your browser.
"""

import io
import os
import base64
import socket
import tempfile

from flask import Flask, render_template, request, jsonify
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "handwriting-lab-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024      # 25 MB upload cap

# ---------------------------------------------------------------------------
# Lazy model cache — load each model only the first time it's needed.
# ---------------------------------------------------------------------------
_cache = {}


def _model(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def digit_model():
    from hlab.digits import load_model
    return _model("digit", load_model)


def letter_model():
    from hlab.letters import load_model
    return _model("letter", load_model)


def calc_model():
    from hlab.calc import load_model
    return _model("calc", load_model)


def signature_model():
    from hlab.signature import load_model
    return _model("signature", load_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_canvas(data_url: str) -> Image.Image:
    """Turn a canvas PNG data-URL (white strokes on black) into a grayscale image."""
    b64 = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return img.convert("L")


def save_upload(file_storage) -> str:
    suffix = os.path.splitext(file_storage.filename)[1] or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file_storage.save(path)
    return path


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def err(message, code=400):
    return jsonify(ok=False, error=message), code


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    """Which models are trained and ready."""
    have = lambda p: os.path.exists(os.path.join("models", p))
    from hlab.ocr import tesseract_available
    from hlab.handwriting_ocr import is_available as handwriting_available
    return jsonify(
        digit=have("digit_cnn.pt"), letter=have("letter_cnn.pt"),
        calc=have("calc_cnn.pt"), signature=have("signature_model.joblib"),
        tesseract=tesseract_available(), handwriting=handwriting_available())


# ---------------------------------------------------------------------------
# Canvas-based features
# ---------------------------------------------------------------------------
@app.post("/api/number")
def api_number():
    try:
        from hlab import digits
        img = decode_canvas(request.json["image"])
        number, results = digits.predict_sequence(digit_model(), img)
    except FileNotFoundError:
        return err("Digit model not trained. Run: python train_digits.py")
    except Exception as e:
        return err(str(e), 500)
    return jsonify(ok=True, number=number,
                   digits=[{"d": d, "conf": round(c, 3), "box": list(b)}
                           for d, c, b in results])


@app.post("/api/letter")
def api_letter():
    try:
        from hlab import letters
        img = decode_canvas(request.json["image"])
        letter, conf, probs = letters.predict(letter_model(), img)
    except FileNotFoundError:
        return err("Letter model not trained. Run: python train_letters.py")
    except Exception as e:
        return err(str(e), 500)
    if letter is None:
        return jsonify(ok=True, letter=None, top=[])
    order = probs.argsort()[::-1][:3]
    top = [{"c": letters.LETTERS[i], "p": round(float(probs[i]), 3)} for i in order]
    return jsonify(ok=True, letter=letter, conf=round(conf, 3), top=top)


@app.post("/api/calc")
def api_calc():
    try:
        from hlab import calc
        img = decode_canvas(request.json["image"])
        out = calc.recognize_expression(calc_model(), img)
    except FileNotFoundError:
        return err("Calculator model not trained. Run: python train_calc.py")
    except Exception as e:
        return err(str(e), 500)
    return jsonify(ok=True, display=out["display"], result=out["result"],
                   error=out["error"], boxes=[list(b) for b in out["boxes"]],
                   tokens=out["tokens"])


# ---------------------------------------------------------------------------
# Upload-based features
# ---------------------------------------------------------------------------
@app.post("/api/ocr")
def api_ocr():
    if "image" not in request.files:
        return err("No image uploaded.")
    mode = request.form.get("mode", "printed")
    path = save_upload(request.files["image"])
    try:
        if mode == "handwriting":
            from hlab import handwriting_ocr
            text = handwriting_ocr.read_handwriting(path)
        else:
            from hlab import ocr
            text = ocr.read_text(path)
    except Exception as e:
        return err(str(e), 500)
    finally:
        os.remove(path)
    return jsonify(ok=True, text=text)


@app.post("/api/signature")
def api_signature():
    genuine = request.files.getlist("genuine")
    questioned = request.files.get("questioned")
    if not genuine or questioned is None:
        return err("Provide genuine signatures and one questioned signature.")
    gpaths = [save_upload(f) for f in genuine]
    qpath = save_upload(questioned)
    try:
        from hlab import signature, report
        prob, verdict = signature.verify(gpaths, qpath, signature_model())
        comp = report.signature_comparison_image(gpaths, qpath)
        explain = report.explanation_text(prob, verdict)
    except FileNotFoundError:
        return err("Signature model not trained. Run: python evaluate_signature.py")
    except Exception as e:
        return err(str(e), 500)
    finally:
        for p in gpaths + [qpath]:
            os.remove(p)
    return jsonify(ok=True, verdict=verdict, probability=round(prob, 4),
                   explanation=explain, comparison=pil_to_data_url(comp))


@app.post("/api/writer/identify")
def api_writer_identify():
    if "image" not in request.files:
        return err("No image uploaded.")
    path = save_upload(request.files["image"])
    try:
        from hlab import writer_id
        name, score, ranked = writer_id.identify_writer(path)
    except FileNotFoundError:
        return err("Signature model not trained. Run: python evaluate_signature.py")
    except Exception as e:
        return err(str(e), 500)
    finally:
        os.remove(path)
    if name is None:
        return err("Nobody enrolled yet. Enrol people first.")
    return jsonify(ok=True, best=name, score=round(score, 3),
                   ranking=[{"name": n, "score": round(s, 3)} for n, s in ranked])


@app.post("/api/writer/enroll")
def api_writer_enroll():
    name = (request.form.get("name") or "").strip()
    files = request.files.getlist("samples")
    if not name or not files:
        return err("Provide a name and at least one sample.")
    paths = [save_upload(f) for f in files]
    try:
        from hlab import writer_id
        n = writer_id.enroll_writer(name, paths)
    except FileNotFoundError:
        return err("Signature model not trained. Run: python evaluate_signature.py")
    except Exception as e:
        return err(str(e), 500)
    finally:
        for p in paths:
            os.remove(p)
    from hlab import writer_id
    return jsonify(ok=True, added=n, enrolled=writer_id.list_writers())


@app.get("/api/writer/list")
def api_writer_list():
    from hlab import writer_id
    return jsonify(ok=True, enrolled=writer_id.list_writers())


@app.post("/api/writer/forget")
def api_writer_forget():
    from hlab import writer_id
    writer_id.forget_all_writers()
    return jsonify(ok=True, enrolled={})


# ---------------------------------------------------------------------------
def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

    print("\n  HandwritingLab web app is running!")
    print(f"  On this PC:        http://localhost:{port}")
    print(f"  On your phone:     http://{_lan_ip()}:{port}   (same Wi-Fi)")
    print("  Press Ctrl+C to stop.\n")
    app.run(host=host, port=port, debug=debug)
