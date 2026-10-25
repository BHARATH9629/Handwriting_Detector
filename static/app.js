"use strict";

// ---- tab switching ----
document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById(t.dataset.tab).classList.add("active");
  });
});

// ---- a drawing pad bound to one <canvas> ----
class Pad {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.clear();
    this.drawing = false;
    this.last = null;
    const pos = e => {
      const r = canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * (canvas.width / r.width),
               y: (e.clientY - r.top) * (canvas.height / r.height) };
    };
    canvas.addEventListener("pointerdown", e => {
      this.drawing = true; this.last = pos(e);
      canvas.setPointerCapture(e.pointerId); e.preventDefault();
    });
    canvas.addEventListener("pointermove", e => {
      if (!this.drawing) return;
      const p = pos(e); this.line(this.last, p); this.last = p; e.preventDefault();
    });
    const stop = () => { this.drawing = false; };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("pointerleave", stop);
  }
  line(a, b) {
    const x = this.ctx;
    x.strokeStyle = "#fff"; x.lineWidth = 16; x.lineCap = "round"; x.lineJoin = "round";
    x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
  }
  clear() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.c.width, this.c.height);
  }
  dataURL() { return this.c.toDataURL("image/png"); }
  drawBoxes(boxes, labels) {
    const x = this.ctx;
    x.strokeStyle = "#39d353"; x.lineWidth = 2;
    x.fillStyle = "#39d353"; x.font = "bold 16px sans-serif";
    boxes.forEach((b, i) => {
      x.strokeRect(b[0], b[1], b[2], b[3]);
      if (labels) x.fillText(String(labels[i]), b[0] + 2, Math.max(12, b[1] - 3));
    });
  }
}

// give every canvas panel its own pad
const pads = {};
document.querySelectorAll(".panel").forEach(panel => {
  const c = panel.querySelector("canvas.pad");
  if (c) pads[panel.id] = new Pad(c);
});

// ---- network helpers ----
async function postJSON(url, obj) {
  const r = await fetch(url, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) });
  return r.json();
}
async function postForm(url, fd) {
  const r = await fetch(url, { method: "POST", body: fd });
  return r.json();
}
const $ = (panel, sel) => document.querySelector("#" + panel + " " + sel);

// ---- actions ----
async function run(action, panel) {
  try {
    if (action === "clear") {
      pads[panel].clear();
      $(panel, ".big").textContent = "";
      $(panel, ".detail").textContent = "";
      return;
    }
    if (action === "number")  return await doNumber(panel);
    if (action === "letter")  return await doLetter(panel);
    if (action === "calc")    return await doCalc(panel);
    if (action === "ocr")     return await doOcr(panel);
    if (action === "signature") return await doSignature(panel);
    if (action.startsWith("writer")) return await doWriter(action, panel);
  } catch (e) {
    alert("Something went wrong: " + e.message);
  }
}

document.querySelectorAll("[data-action]").forEach(btn => {
  const panel = btn.closest(".panel").id;
  btn.addEventListener("click", () => run(btn.dataset.action, panel));
});

function busy(panel, msg) {
  const b = $(panel, ".big"); if (b) b.textContent = "";
  const d = $(panel, ".detail"); if (d) { d.className = "detail spin"; d.textContent = msg; }
}

async function doNumber(panel) {
  busy(panel, "thinking…");
  const res = await postJSON("/api/number", { image: pads[panel].dataURL() });
  if (!res.ok) return fail(panel, res.error);
  $(panel, ".big").textContent = res.number || "—";
  if (res.digits.length) {
    pads[panel].drawBoxes(res.digits.map(x => x.box), res.digits.map(x => x.d));
    $(panel, ".detail").className = "detail";
    $(panel, ".detail").textContent = res.digits
      .map(x => `  ${x.d}  (${Math.round(x.conf * 100)}%)`).join("\n");
  } else $(panel, ".detail").textContent = "nothing drawn yet";
}

async function doLetter(panel) {
  busy(panel, "thinking…");
  const res = await postJSON("/api/letter", { image: pads[panel].dataURL() });
  if (!res.ok) return fail(panel, res.error);
  $(panel, ".big").textContent = res.letter || "—";
  $(panel, ".detail").className = "detail";
  $(panel, ".detail").textContent = res.letter
    ? "top guesses:\n" + res.top.map(t => `  ${t.c}: ${Math.round(t.p * 100)}%`).join("\n")
    : "nothing drawn yet";
}

async function doCalc(panel) {
  busy(panel, "thinking…");
  const res = await postJSON("/api/calc", { image: pads[panel].dataURL() });
  if (!res.ok) return fail(panel, res.error);
  if (res.boxes.length) pads[panel].drawBoxes(res.boxes, res.tokens);
  $(panel, ".detail").className = "detail";
  if (res.error) {
    $(panel, ".big").textContent = "🤔";
    $(panel, ".detail").textContent = `read: ${res.display || "—"}\n(${res.error})`;
  } else {
    $(panel, ".big").textContent = `${res.display} = ${res.result}`;
    $(panel, ".detail").textContent = `read these symbols: ${res.display}`;
  }
}

async function doOcr(panel) {
  const f = $(panel, ".file").files[0];
  if (!f) return alert("Choose an image first.");
  const mode = (document.querySelector('input[name="ocrmode"]:checked') || {}).value || "printed";
  $(panel, ".ocrtext").textContent =
    mode === "handwriting" ? "reading handwriting… (first time downloads the AI model)" : "reading…";
  const fd = new FormData(); fd.append("image", f); fd.append("mode", mode);
  const res = await postForm("/api/ocr", fd);
  $(panel, ".ocrtext").textContent = res.ok ? res.text : "Error: " + res.error;
}

async function doSignature(panel) {
  const gen = $(panel, ".genuine").files;
  const q = $(panel, ".questioned").files[0];
  if (!gen.length || !q) return alert("Choose genuine signatures and a questioned one.");
  $(panel, ".verdict").textContent = "checking…";
  $(panel, ".detail").textContent = "";
  $(panel, ".comparison").style.display = "none";
  const fd = new FormData();
  for (const g of gen) fd.append("genuine", g);
  fd.append("questioned", q);
  const res = await postForm("/api/signature", fd);
  if (!res.ok) { $(panel, ".verdict").textContent = ""; return alert("Error: " + res.error); }
  const v = $(panel, ".verdict");
  v.textContent = res.verdict;
  v.className = "verdict big " +
    (res.verdict.includes("GENUINE") ? "genuine" :
     res.verdict.includes("BORDERLINE") ? "borderline" : "forgery");
  $(panel, ".detail").textContent = res.explanation;
  const img = $(panel, ".comparison");
  img.src = res.comparison; img.style.display = "block";
}

async function doWriter(action, panel) {
  if (action === "writer-enroll") {
    const name = $(panel, ".wname").value.trim();
    const files = $(panel, ".wsamples").files;
    if (!name || !files.length) return alert("Enter a name and pick samples.");
    const fd = new FormData(); fd.append("name", name);
    for (const f of files) fd.append("samples", f);
    const res = await postForm("/api/writer/enroll", fd);
    if (!res.ok) return alert("Error: " + res.error);
    renderEnrolled(panel, res.enrolled);
  } else if (action === "writer-forget") {
    const res = await postForm("/api/writer/forget", new FormData());
    renderEnrolled(panel, res.enrolled);
    $(panel, ".wresult").textContent = "";
  } else if (action === "writer-identify") {
    const f = $(panel, ".wquery").files[0];
    if (!f) return alert("Pick a sample to identify.");
    $(panel, ".wresult").textContent = "identifying…";
    const fd = new FormData(); fd.append("image", f);
    const res = await postForm("/api/writer/identify", fd);
    if (!res.ok) { $(panel, ".wresult").textContent = res.error; return; }
    $(panel, ".wresult").textContent =
      `Best match: ${res.best} (${Math.round(res.score * 100)}%)\n\nranking:\n` +
      res.ranking.map(r => `  ${r.name}: ${Math.round(r.score * 100)}%`).join("\n");
  }
}

function renderEnrolled(panel, enrolled) {
  const names = Object.entries(enrolled || {});
  $(panel, ".enrolled").textContent = names.length
    ? "Enrolled:\n" + names.map(([n, c]) => `  • ${n}: ${c} sample(s)`).join("\n")
    : "Enrolled: (nobody yet)";
}

function fail(panel, msg) {
  $(panel, ".big").textContent = "⚠";
  $(panel, ".detail").className = "detail";
  $(panel, ".detail").textContent = msg || "error";
}

// ---- on load: show model/engine status ----
(async () => {
  try {
    const s = await (await fetch("/api/status")).json();
    document.getElementById("ocr-status").textContent =
      s.tesseract ? "(Tesseract ready ✓)" : "(needs Tesseract — see README)";
    document.getElementById("sig-status").textContent =
      s.signature ? "" : "(model not trained)";
    const w = document.querySelector("#writer");
    if (w) {
      const r = await (await fetch("/api/writer/list")).json();
      renderEnrolled("writer", r.enrolled);
    }
  } catch (e) { /* ignore */ }
})();
