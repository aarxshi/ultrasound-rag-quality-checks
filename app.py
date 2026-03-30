import os
import random
import json
import numpy as np
from PIL import Image
import gradio as gr

# ── RAG (pure Python, no torch/TF needed) ────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

QUALITY_BUCKETS = {
    "depth": ["depth"],
    "gain": ["gain"],
    "alignment": ["beam", "aligned", "perpendicular"],
    "coverage": ["required structures", "visible"],
    "artifact": ["shadow", "artifact", "noise"]
}

EXPLANATIONS = {
    "depth": "Image depth does not fully capture the target anatomy.",
    "gain": "Gain settings reduce contrast or obscure structures.",
    "alignment": "Probe alignment causes distortion or foreshortening.",
    "coverage": "Required structures are partially missing.",
    "artifact": "Artifacts or shadowing reduce interpretability."
}

def simple_rag(question, top_k=3):
    """Keyword overlap RAG — no embeddings needed."""
    question_words = set(question.lower().split())
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk["text"].lower().split())
        score = len(question_words & chunk_words)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks[i]["text"] for _, i in scores[:top_k]]

def get_explanations():
    results = simple_rag("Why is this ultrasound scan low quality?")
    found = set()
    for text in results:
        t = text.lower()
        for label, keywords in QUALITY_BUCKETS.items():
            if any(k in t for k in keywords):
                found.add(label)
    return [EXPLANATIONS[f] for f in found]

# ── Dummy classifier ──────────────────────────────────────────────────────────

def classify(image):
    """Dummy classifier — replace with real model later."""
    r = random.random()
    if r > 0.65:
        return "Unclear scan", "High", "Retake the scan. Adjust probe angle or gain."
    elif r < 0.35:
        return "Clear scan", "High", "Scan quality acceptable. Proceed with interpretation."
    else:
        return "Uncertain scan quality", "Moderate", "Consider retaking the scan."

# ── UI helpers ────────────────────────────────────────────────────────────────

def badge_style(result):
    return {
        "Clear scan":            "background:#EAF3DE;color:#3B6D11",
        "Unclear scan":          "background:#FCEBEB;color:#A32D2D",
        "Uncertain scan quality":"background:#FAEEDA;color:#854F0B",
    }.get(result, "background:#f0f0f0;color:#666")

def dot_color(result):
    return {
        "Clear scan":            "#639922",
        "Unclear scan":          "#E24B4A",
        "Uncertain scan quality":"#BA7517",
    }.get(result, "#aaa")

def conf_pct(confidence):
    return {"High": 90, "Moderate": 55, "Low": 30}.get(confidence, 0)

def render(result, confidence, guidance, explanations):
    bs = badge_style(result)
    dc = dot_color(result)
    pct = conf_pct(confidence)

    items_html = ""
    if explanations:
        items_html = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;font-size:13px;'
            f'line-height:1.5;margin-bottom:6px">'
            f'<span style="width:5px;height:5px;border-radius:50%;background:#378ADD;'
            f'flex-shrink:0;margin-top:6px;display:inline-block"></span><span style="color:#111!important">{e}</span></div>'
            for e in explanations
        )
        explain_section = f"""
        <div style="border-top:0.5px solid #e0e0e0;padding-top:14px;margin-top:4px">
          <div style="font-size:12px;color:#888;margin-bottom:8px">Quality notes</div>
          {items_html}
        </div>"""
    else:
        explain_section = ""

    return f"""
    <div style="font-family:system-ui,sans-serif;display:flex;flex-direction:column;gap:16px;padding:4px 0;opacity:1!important">
      <div>
        <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#999!important;margin-bottom:8px">Result</div>
        <span style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:8px;font-size:13px;font-weight:500;{bs}">
          <span style="width:7px;height:7px;border-radius:50%;background:{dc};flex-shrink:0;display:inline-block"></span>
          {result}
        </span>
      </div>
      <div style="border-top:0.5px solid #e0e0e0;padding-top:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
          <span style="font-size:12px;color:#888!important">Confidence</span>
          <span style="font-size:13px;font-weight:500;color:#111!important">{confidence}</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:12px;color:#999!important">Low</span>
          <div style="flex:1;height:4px;border-radius:2px;background:#eee;overflow:hidden">
            <div style="height:100%;width:{pct}%;border-radius:2px;background:#185FA5"></div>
          </div>
          <span style="font-size:12px;color:#999!important">High</span>
        </div>
      </div>
      <div style="border-top:0.5px solid #e0e0e0;padding-top:14px">
        <div style="font-size:12px;color:#888!important;margin-bottom:6px">Recommendation</div>
        <div style="background:#f7f7f5;border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.5;color:#111!important">
          {guidance}
        </div>
      </div>
      {explain_section}
    </div>
    """

def predict(image):
    if image is None:
        return render("—", "—", "Please upload a scan.", [])
    result, confidence, guidance = classify(image)
    explanations = get_explanations() if result != "Clear scan" else []
    return render(result, confidence, guidance, explanations)

# ── Gradio app ────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto; }
.panel-card { background: white; border: 0.5px solid #e8e8e8; border-radius: 12px; padding: 20px; }
footer { display: none !important; }
.gradio-container div { color: unset !important; opacity: 1 !important; }
"""

with gr.Blocks(title="Ultrasound Scan Quality") as demo:
    gr.HTML("""
    <div style="padding:8px 0 20px;border-bottom:0.5px solid #e8e8e8;margin-bottom:20px">
      <h1 style="font-size:17px;font-weight:500;margin:0;color:#111">Ultrasound scan quality</h1>
      <p style="font-size:13px;color:#888;margin:4px 0 0">Flags unclear scans at capture time</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="panel-card"):
            gr.HTML('<p style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#999;margin:0 0 10px">Input</p>')
            image_input = gr.Image(type="pil", label="", height=260, show_label=False)
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear", size="sm")
                submit_btn = gr.Button("Analyse scan", variant="primary", size="sm")

        with gr.Column(elem_classes="panel-card"):
            gr.HTML('<p style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#999;margin:0 0 10px">Assessment</p>')
            output_html = gr.HTML(
                value='<div style="font-family:system-ui,sans-serif;padding:4px 0;color:#aaa;font-size:13px">Upload a scan and click Analyse.</div>'
            )

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_html)
    clear_btn.add(image_input)
    clear_btn.click(
        fn=lambda: '<div style="font-family:system-ui,sans-serif;padding:4px 0;color:#aaa;font-size:13px">Upload a scan and click Analyse.</div>',
        outputs=output_html
    )

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, css=CSS)
