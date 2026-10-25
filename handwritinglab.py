"""
HandwritingLab — desktop app with handwriting tools:

    1. Draw Numbers    — read one or many hand-drawn digits (CNN, MNIST ~99%)
    2. Draw a Letter   — recognise a hand-drawn letter (CNN, EMNIST balanced)
    3. Calculator      — draw a sum like 7+3 and get the answer (CNN)
    4. Read Text       — OCR via Tesseract
    5. Who Wrote It?   — writer identification by handwriting style
    6. Signature Check — forgery detection with a visual explanation (CEDAR)

Run with:   python handwritinglab.py

Heavy models load lazily (only when you first use a tab), so the window opens
instantly. The algorithms live in the `hlab` package; this file is just the UI.
"""

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk

IMG_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]


def _error_box(title, exc):
    messagebox.showerror(title, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
class Preview(ttk.Label):
    """Shows an image file or a PIL image, keeping a reference so it doesn't vanish."""

    def __init__(self, parent, size=150):
        super().__init__(parent, relief="solid", borderwidth=1, anchor="center")
        self._size = size
        self._photo = None

    def show_path(self, path):
        try:
            self.show_image(Image.open(path))
        except Exception:
            pass

    def show_image(self, pil_img, max_w=None):
        img = pil_img.convert("RGB")
        if max_w:
            if img.width > max_w:
                img = img.resize((max_w, int(img.height * max_w / img.width)))
        else:
            img.thumbnail((self._size, self._size))
        self._photo = ImageTk.PhotoImage(img)
        self.config(image=self._photo)


class DrawingPad(ttk.Frame):
    """A black canvas you draw white strokes on, mirrored into a Pillow image."""

    SIZE, PEN = 280, 18

    def __init__(self, parent):
        super().__init__(parent)
        self._last = None
        self._overlays = []
        self.canvas = tk.Canvas(self, width=self.SIZE, height=self.SIZE, bg="black",
                                highlightthickness=1, highlightbackground="#888",
                                cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_last", None))
        self._img = Image.new("L", (self.SIZE, self.SIZE), 0)
        self._pen = ImageDraw.Draw(self._img)

    def _start(self, e):
        self._clear_overlays()
        self._last = (e.x, e.y)
        self._draw(e)

    def _draw(self, e):
        x, y = e.x, e.y
        x0, y0 = self._last or (x, y)
        self.canvas.create_line(x0, y0, x, y, fill="white", width=self.PEN,
                                capstyle=tk.ROUND, smooth=True)
        self._pen.line([x0, y0, x, y], fill=255, width=self.PEN)
        r = self.PEN // 2
        self._pen.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self._last = (x, y)

    def _clear_overlays(self):
        for o in self._overlays:
            self.canvas.delete(o)
        self._overlays = []

    def clear(self):
        self.canvas.delete("all")
        self._overlays = []
        self._img = Image.new("L", (self.SIZE, self.SIZE), 0)
        self._pen = ImageDraw.Draw(self._img)

    def image(self):
        return self._img.copy()

    def highlight(self, boxes, labels=None):
        """Draw green rectangles (and optional labels) over detected ink blobs."""
        self._clear_overlays()
        for i, (x, y, w, h) in enumerate(boxes):
            self._overlays.append(self.canvas.create_rectangle(
                x, y, x + w, y + h, outline="#39d353", width=2))
            if labels:
                self._overlays.append(self.canvas.create_text(
                    x + 2, y - 8, text=str(labels[i]), fill="#39d353",
                    anchor="w", font=("Segoe UI", 11, "bold")))


# ---------------------------------------------------------------------------
# Tab 1 — Draw numbers (one or many digits)
# ---------------------------------------------------------------------------
class DigitTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self._model = None

        ttk.Label(self, text="Draw one or more digits (leave gaps between them).",
                  font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2,
                                              sticky="w", pady=(0, 8))
        self.pad = DrawingPad(self)
        self.pad.grid(row=1, column=0, rowspan=4, padx=(0, 14))

        self.result = ttk.Label(self, text="", font=("Segoe UI", 30, "bold"))
        self.result.grid(row=1, column=1, sticky="w")
        self.detail = ttk.Label(self, text="", foreground="#555",
                                font=("Consolas", 9), justify="left")
        self.detail.grid(row=2, column=1, sticky="nw")
        ttk.Label(self, text="what the computer sees:", foreground="#777",
                  font=("Segoe UI", 8)).grid(row=3, column=1, sticky="sw")
        self.seen = Preview(self, size=80)
        self.seen.grid(row=4, column=1, sticky="nw")

        btns = ttk.Frame(self)
        btns.grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Recognise", command=self._recognise).pack(
            side="left", padx=(0, 6))
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left")

    def _clear(self):
        self.pad.clear()
        self.result.config(text="")
        self.detail.config(text="")
        self.seen.config(image="")

    def _recognise(self):
        try:
            if self._model is None:
                self.detail.config(text="loading model…")
                self.update_idletasks()
                from hlab import digits
                self._model = digits.load_model()
            from hlab import digits, report
            number, results = digits.predict_sequence(self._model, self.pad.image())
        except FileNotFoundError:
            messagebox.showinfo("Model missing",
                                "Train the digit model once:\n\npython train_digits.py")
            return
        except Exception as exc:
            _error_box("Recognition failed", exc)
            return

        if not results:
            self.result.config(text="—")
            self.detail.config(text="nothing drawn yet")
            self.seen.config(image="")
            return
        boxes = [r[2] for r in results]
        self.pad.highlight(boxes, labels=[r[0] for r in results])
        self.result.config(text=number)
        self.detail.config(text="\n".join(f"  {d}  ({c*100:.0f}%)"
                                          for d, c, _ in results))
        strip = report.digit_view_strip(self.pad.image(), boxes)
        if strip is not None:
            self.seen.show_image(strip)


# ---------------------------------------------------------------------------
# Tab 2 — Draw a letter
# ---------------------------------------------------------------------------
class LetterTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self._model = None

        ttk.Label(self, text="Draw a single capital letter (A-Z), then Recognise.",
                  font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2,
                                              sticky="w", pady=(0, 8))
        self.pad = DrawingPad(self)
        self.pad.grid(row=1, column=0, rowspan=3, padx=(0, 14))

        self.result = ttk.Label(self, text="", font=("Segoe UI", 30, "bold"))
        self.result.grid(row=1, column=1, sticky="w")
        self.detail = ttk.Label(self, text="", foreground="#555",
                                font=("Consolas", 9), justify="left")
        self.detail.grid(row=2, column=1, sticky="nw")

        btns = ttk.Frame(self)
        btns.grid(row=3, column=1, sticky="sw")
        ttk.Button(btns, text="Recognise", command=self._recognise).pack(
            side="left", padx=(0, 6))
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left")

    def _clear(self):
        self.pad.clear()
        self.result.config(text="")
        self.detail.config(text="")

    def _recognise(self):
        try:
            if self._model is None:
                self.detail.config(text="loading model…")
                self.update_idletasks()
                from hlab import letters
                self._model = letters.load_model()
            from hlab import letters
            letter, conf, probs = letters.predict(self._model, self.pad.image())
        except FileNotFoundError:
            messagebox.showinfo("Letters not ready",
                                "The letter model isn't trained yet.\n\n"
                                "Train it once:\n\npython train_letters.py")
            self.detail.config(text="")
            return
        except Exception as exc:
            _error_box("Recognition failed", exc)
            return
        if letter is None:
            self.result.config(text="—")
            self.detail.config(text="nothing drawn yet")
            return
        self.result.config(text=letter)
        order = probs.argsort()[::-1][:3]
        self.detail.config(text="top guesses:\n" + "\n".join(
            f"  {letters.LETTERS[i]}: {probs[i]*100:4.0f}%" for i in order))


# ---------------------------------------------------------------------------
# Tab — Handwriting calculator
# ---------------------------------------------------------------------------
class CalcTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self._model = None

        ttk.Label(self, text="Draw a sum, e.g.  7+3 . Use  +  -  × (cross)  / (slash).",
                  font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2,
                                              sticky="w")
        ttk.Label(self, text="Leave small gaps between symbols.",
                  foreground="#777", font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.pad = DrawingPad(self)
        self.pad.grid(row=2, column=0, rowspan=3, padx=(0, 14))

        self.result = ttk.Label(self, text="", font=("Segoe UI", 24, "bold"))
        self.result.grid(row=2, column=1, sticky="w")
        self.detail = ttk.Label(self, text="", foreground="#555",
                                font=("Consolas", 10), justify="left")
        self.detail.grid(row=3, column=1, sticky="nw")

        btns = ttk.Frame(self)
        btns.grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Calculate", command=self._calc).pack(
            side="left", padx=(0, 6))
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left")

    def _clear(self):
        self.pad.clear()
        self.result.config(text="")
        self.detail.config(text="")

    def _calc(self):
        try:
            if self._model is None:
                self.detail.config(text="loading model…")
                self.update_idletasks()
                from hlab import calc
                self._model = calc.load_model()
            from hlab import calc
            out = calc.recognize_expression(self._model, self.pad.image())
        except FileNotFoundError:
            messagebox.showinfo("Calculator not ready",
                                "Train the calculator model once:\n\npython train_calc.py")
            self.detail.config(text="")
            return
        except Exception as exc:
            _error_box("Calculation failed", exc)
            return

        self.pad.highlight(out["boxes"], labels=out["tokens"])
        if out["error"]:
            self.result.config(text="🤔", foreground="#9a6700")
            self.detail.config(text=f"read: {out['display'] or '—'}\n({out['error']})")
        else:
            self.result.config(text=f"{out['display']} = {out['result']}",
                               foreground="#1a7f37")
            self.detail.config(text=f"read these symbols: {out['display']}")


# ---------------------------------------------------------------------------
# Tab 3 — OCR
# ---------------------------------------------------------------------------
class OcrTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(4, weight=1)

        from hlab import ocr
        ok = ocr.tesseract_available()
        ttk.Label(self, text="Load an image of text and extract it.",
                  font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, foreground=("#1a7f37" if ok else "#b42318"),
                  text=("Tesseract engine: detected ✓" if ok else
                        "Tesseract engine: not installed — see README.")
                  ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))

        self.mode = tk.StringVar(value="printed")
        modes = ttk.Frame(self)
        modes.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Radiobutton(modes, text="Printed / typed (fast)", value="printed",
                        variable=self.mode).pack(side="left")
        ttk.Radiobutton(modes, text="Handwriting (AI, slower)", value="handwriting",
                        variable=self.mode).pack(side="left", padx=10)

        ttk.Button(self, text="Open image…", command=self._open).grid(
            row=3, column=0, sticky="w")
        self.preview = Preview(self)
        self.preview.grid(row=4, column=0, sticky="n", pady=(8, 0))
        self.text = tk.Text(self, wrap="word", width=44, height=15)
        self.text.grid(row=3, column=1, rowspan=2, sticky="nsew", padx=(12, 0))

    def _open(self):
        path = filedialog.askopenfilename(title="Choose an image", filetypes=IMG_TYPES)
        if not path:
            return
        self.preview.show_path(path)
        handwriting = self.mode.get() == "handwriting"
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "Reading handwriting… (first time downloads the AI model)"
                                if handwriting else "Reading…")
        self.update_idletasks()
        try:
            if handwriting:
                from hlab import handwriting_ocr
                result = handwriting_ocr.read_handwriting(path)
            else:
                from hlab import ocr
                result = ocr.read_text(path)
        except Exception as exc:
            _error_box("OCR failed", exc)
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", result)


# ---------------------------------------------------------------------------
# Tab 4 — Who wrote it?
# ---------------------------------------------------------------------------
class WriterTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="Step 1 — enrol people (several samples each).",
                  font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2, sticky="w")
        row = ttk.Frame(self)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(row, text="Name:").pack(side="left")
        self.name = ttk.Entry(row, width=18)
        self.name.pack(side="left", padx=6)
        ttk.Button(row, text="Add samples…", command=self._add).pack(side="left")
        ttk.Button(row, text="Forget all", command=self._forget).pack(side="left", padx=6)

        self.enrolled = ttk.Label(self, text="", justify="left", foreground="#333")
        self.enrolled.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._refresh()

        ttk.Separator(self, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(self, text="Step 2 — identify an unknown sample.",
                  font=("Segoe UI", 11)).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(self, text="Identify a sample…", command=self._identify).grid(
            row=5, column=0, sticky="w", pady=6)
        self.preview = Preview(self)
        self.preview.grid(row=6, column=0, sticky="nw")
        self.result = ttk.Label(self, text="", justify="left", font=("Consolas", 10))
        self.result.grid(row=6, column=1, sticky="nw", padx=10)

    def _refresh(self):
        from hlab import writer_id
        people = writer_id.list_writers()
        txt = ("Enrolled:\n" + "\n".join(f"  • {n}: {c} sample(s)"
                                         for n, c in people.items())
               if people else "Enrolled: (nobody yet)")
        self.enrolled.config(text=txt)

    def _add(self):
        name = self.name.get().strip()
        if not name:
            messagebox.showinfo("Name needed", "Type a name first.")
            return
        paths = filedialog.askopenfilenames(
            title=f"Pick handwriting samples for {name}", filetypes=IMG_TYPES)
        if not paths:
            return
        try:
            from hlab import writer_id
            n = writer_id.enroll_writer(name, list(paths))
        except FileNotFoundError:
            messagebox.showinfo("Model missing", "Train the signature model once:"
                                                 "\n\npython evaluate_signature.py")
            return
        except Exception as exc:
            _error_box("Enrolment failed", exc)
            return
        self._refresh()
        messagebox.showinfo("Added", f"Added {n} sample(s) for {name}.")

    def _forget(self):
        if messagebox.askyesno("Forget all", "Remove everyone enrolled?"):
            from hlab import writer_id
            writer_id.forget_all_writers()
            self._refresh()

    def _identify(self):
        path = filedialog.askopenfilename(title="Pick an unknown sample",
                                          filetypes=IMG_TYPES)
        if not path:
            return
        try:
            from hlab import writer_id
            name, score, ranked = writer_id.identify_writer(path)
        except Exception as exc:
            _error_box("Identification failed", exc)
            return
        if name is None:
            messagebox.showinfo("Nobody enrolled", "Enrol some people first.")
            return
        self.preview.show_path(path)
        lines = [f"Best match: {name}  ({score*100:.0f}%)", "", "ranking:"]
        lines += [f"  {n:12s} {s*100:5.0f}%" for n, s in ranked]
        self.result.config(text="\n".join(lines))


# ---------------------------------------------------------------------------
# Tab 5 — Signature check (with visual explanation + report)
# ---------------------------------------------------------------------------
class SignatureTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self._genuine, self._questioned = [], None
        self._last = None                     # (prob, verdict) for saving a report

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Compare a signature against known-genuine ones.",
                  font=("Segoe UI", 11)).pack(anchor="w")
        self.accuracy = ttk.Label(top, text="", foreground="#555",
                                  font=("Segoe UI", 8))
        self.accuracy.pack(anchor="w", pady=(0, 6))
        self._show_accuracy()

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Button(controls, text="Genuine signatures…",
                   command=self._pick_genuine).grid(row=0, column=0, sticky="w")
        self.gl = ttk.Label(controls, text="none", foreground="#555")
        self.gl.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(controls, text="Questioned signature…",
                   command=self._pick_questioned).grid(row=0, column=2, sticky="w",
                                                       padx=(12, 0))
        self.ql = ttk.Label(controls, text="none", foreground="#555")
        self.ql.grid(row=0, column=3, sticky="w", padx=6)

        action = ttk.Frame(self)
        action.pack(fill="x", pady=8)
        ttk.Button(action, text="Check signature", command=self._check).pack(side="left")
        self.save_btn = ttk.Button(action, text="Save report",
                                   command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=8)

        self.result = ttk.Label(self, text="", font=("Segoe UI", 16, "bold"))
        self.result.pack(anchor="w")
        self.detail = ttk.Label(self, text="", foreground="#444", wraplength=600,
                                justify="left")
        self.detail.pack(anchor="w")
        self.compare = Preview(self)
        self.compare.pack(anchor="w", pady=(8, 0))

    def _show_accuracy(self):
        try:
            from hlab.signature import load_model
            m = load_model().get("metrics", {})
            self.accuracy.config(
                text=f"Tested on unseen signers — AUC {m['auc']*100:.0f}%, "
                     f"Equal-Error-Rate {m['eer']*100:.0f}% (lower = better).")
        except Exception:
            self.accuracy.config(text="Signature model not trained — run "
                                      "evaluate_signature.py")

    def _pick_genuine(self):
        paths = filedialog.askopenfilenames(
            title="Pick one or more genuine signatures", filetypes=IMG_TYPES)
        if paths:
            self._genuine = list(paths)
            self.gl.config(text=f"{len(self._genuine)} chosen")

    def _pick_questioned(self):
        path = filedialog.askopenfilename(title="Pick the signature to check",
                                          filetypes=IMG_TYPES)
        if path:
            self._questioned = path
            self.ql.config(text=os.path.basename(path))

    def _check(self):
        if not self._genuine or not self._questioned:
            messagebox.showinfo("Need both", "Choose genuine signatures and a "
                                             "questioned one first.")
            return
        try:
            from hlab import signature, report
            prob, verdict = signature.verify(self._genuine, self._questioned)
            comp = report.signature_comparison_image(self._genuine, self._questioned)
            explain = report.explanation_text(prob, verdict)
        except FileNotFoundError:
            messagebox.showinfo("Model missing", "Train the signature model once:"
                                                 "\n\npython evaluate_signature.py")
            return
        except Exception as exc:
            _error_box("Signature check failed", exc)
            return
        color = {"LIKELY GENUINE": "#1a7f37",
                 "BORDERLINE — check manually": "#9a6700",
                 "LIKELY FORGERY": "#b42318"}.get(verdict, "#333")
        self.result.config(text=verdict, foreground=color)
        self.detail.config(text=explain)
        self.compare.show_image(comp, max_w=600)
        self._last = (prob, verdict)
        self.save_btn.config(state="normal")

    def _save(self):
        if not self._last:
            return
        try:
            from hlab import report
            prob, verdict = self._last
            png = report.save_signature_report(self._genuine, self._questioned,
                                               prob, verdict)
        except Exception as exc:
            _error_box("Saving report failed", exc)
            return
        folder = os.path.dirname(os.path.abspath(png))
        if messagebox.askyesno("Report saved",
                               f"Saved to:\n{png}\n\nOpen the reports folder?"):
            try:
                os.startfile(folder)
            except Exception:
                pass


# ---------------------------------------------------------------------------
def main():
    app = tk.Tk()
    app.title("HandwritingLab")
    app.geometry("720x620")
    app.minsize(680, 600)
    nb = ttk.Notebook(app)
    nb.pack(fill="both", expand=True, padx=8, pady=8)
    nb.add(DigitTab(nb), text="✍ Draw Numbers")
    nb.add(LetterTab(nb), text="🔤 Draw a Letter")
    nb.add(CalcTab(nb), text="🧮 Calculator")
    nb.add(OcrTab(nb), text="📄 Read Text")
    nb.add(WriterTab(nb), text="🧑 Who Wrote It?")
    nb.add(SignatureTab(nb), text="✒ Signature Check")
    app.mainloop()


if __name__ == "__main__":
    main()
