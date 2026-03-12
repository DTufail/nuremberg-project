"""
app.py  —  Nuremberg Scholar UI (HuggingFace Spaces)
======================================================
Designed for Gradio >=5.9.0 on HuggingFace Spaces.
Pipeline logic stays in rag.py.
"""

from rag import NurembergScholar, GROQ_MODEL
import gradio as gr

# ── Initialise pipeline ───────────────────────────────────────────────────────

scholar = NurembergScholar()

# ── Theme ─────────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("DM Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#fafafa",
    body_text_color="#18181b",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e4e4e7",
    block_radius="12px",
    block_shadow="0 1px 2px 0 rgba(0,0,0,.03)",
    block_label_text_size="*text_xs",
    input_background_fill="#ffffff",
    input_border_color="#d4d4d8",
    input_border_width="1px",
    input_radius="10px",
    input_shadow="none",
    input_text_size="*text_sm",
    button_primary_background_fill="#b91c1c",
    button_primary_background_fill_hover="#991b1b",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_primary_shadow="0 1px 2px 0 rgba(0,0,0,.05)",
    button_large_radius="10px",
    button_large_text_size="*text_sm",
    button_large_text_weight="600",
    # dark mode
    body_background_fill_dark="#09090b",
    body_text_color_dark="#fafafa",
    block_background_fill_dark="#18181b",
    block_border_color_dark="#27272a",
    input_background_fill_dark="#18181b",
    input_border_color_dark="#3f3f46",
    button_primary_background_fill_dark="#dc2626",
    button_primary_background_fill_hover_dark="#b91c1c",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
}
footer { display: none !important; }

/* Hero */
.ns-hero {
    text-align: center;
    padding: 2rem 1rem 0.4rem;
}
.ns-hero h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #18181b;
    margin: 0 0 0.25rem 0;
}
.ns-hero p {
    font-size: 0.875rem;
    color: #71717a;
    margin: 0 0 0.5rem 0;
}
.ns-hero .ns-pills {
    display: flex;
    justify-content: center;
    gap: 0.375rem;
    flex-wrap: wrap;
}
.ns-hero .ns-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 9999px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: #a1a1aa;
    letter-spacing: 0.01em;
}

/* Query box — CRITICAL: keep label visible but restyle it */
#ns-query {
    margin-bottom: 0.25rem !important;
}
#ns-query label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #a1a1aa !important;
    margin-bottom: 0.25rem !important;
}
#ns-query textarea {
    font-size: 0.9rem !important;
    padding: 0.7rem 0.85rem !important;
    min-height: 42px !important;
    line-height: 1.4 !important;
    border-color: #d4d4d8 !important;
}
#ns-query textarea:focus {
    border-color: #b91c1c !important;
    box-shadow: 0 0 0 2px rgba(185,28,28,0.08) !important;
}

/* Search button */
#ns-search {
    min-height: 42px !important;
    min-width: 90px !important;
}

/* Search row alignment */
#ns-search-row {
    gap: 0.5rem !important;
    align-items: flex-end !important;
}

/* Examples — compact pills */
#ns-examples {
    margin: 0.25rem 0 0.75rem 0 !important;
    padding: 0 !important;
}
#ns-examples .gallery {
    gap: 0.25rem !important;
}
#ns-examples button {
    border-radius: 9999px !important;
    font-size: 0.72rem !important;
    padding: 0.25rem 0.65rem !important;
    border: 1px solid #e4e4e7 !important;
    background: #fafafa !important;
    color: #71717a !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 320px !important;
}
#ns-examples button:hover {
    border-color: #fecaca !important;
    background: #fef2f2 !important;
    color: #b91c1c !important;
}

/* Source cards */
.ns-cards {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    padding: 0.15rem 0;
}
.ns-c {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.5rem;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    font-size: 0.72rem;
    color: #52525b;
    line-height: 1.3;
    transition: all 0.12s;
}
.ns-c:hover {
    border-color: #fecaca;
    background: #fef2f2;
}
.ns-n {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 16px; height: 16px;
    background: #b91c1c;
    color: #fff;
    border-radius: 4px;
    font-size: 0.58rem;
    font-weight: 700;
}
.ns-m {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: #a1a1aa;
}

/* Answer */
#ns-answer {
    min-height: 40px;
    padding: 1rem 1.15rem !important;
}
#ns-answer p, #ns-answer li {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}

/* Citation panel */
#ns-cite {
    background: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    padding: 0.6rem 0.85rem !important;
}
#ns-cite p {
    font-size: 0.78rem !important;
    color: #71717a !important;
    line-height: 1.45 !important;
    margin: 0 !important;
}
#ns-cite code {
    font-size: 0.68rem !important;
    background: #f4f4f5 !important;
    padding: 0.08rem 0.25rem !important;
    border-radius: 3px !important;
}
#ns-cite strong { color: #3f3f46 !important; }

/* Accordion */
#ns-detail .label-wrap {
    font-size: 0.78rem !important;
    color: #a1a1aa !important;
}
#ns-detail p,
#ns-detail span,
#ns-detail li,
#ns-detail strong,
#ns-detail em {
    color: #3f3f46 !important;
}
#ns-detail strong {
    font-weight: 600 !important;
    color: #18181b !important;
}
#ns-detail code {
    color: #52525b !important;
    background: #f4f4f5 !important;
    padding: 0.08rem 0.25rem !important;
    border-radius: 3px !important;
    font-size: 0.72rem !important;
}
#ns-detail hr {
    border-color: #e4e4e7 !important;
    margin: 0.75rem 0 !important;
}
#ns-detail blockquote {
    border-left: 2px solid #fecaca !important;
    background: #fef2f2 !important;
    padding: 0.4rem 0.75rem !important;
    margin: 0.3rem 0 !important;
    border-radius: 0 6px 6px 0 !important;
    font-size: 0.82rem !important;
    color: #52525b !important;
}

/* Footer */
.ns-ft {
    text-align: center;
    padding: 1rem 0 0.5rem;
    margin-top: 0.5rem;
    border-top: 1px solid #e4e4e7;
}
.ns-ft p {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: #a1a1aa;
    margin: 0;
}
.ns-ft a { color: #b91c1c; text-decoration: none; }
.ns-ft a:hover { text-decoration: underline; }

/* Dark mode overrides for custom HTML */
@media (prefers-color-scheme: dark) {
    .ns-hero h1 { color: #fafafa; }
    .ns-hero p { color: #a1a1aa; }
    .ns-hero .ns-pill { background: #27272a; border-color: #3f3f46; color: #71717a; }
    .ns-c { background: #27272a; border-color: #3f3f46; color: #d4d4d8; }
    .ns-c:hover { border-color: #7f1d1d; background: #1c0a0a; }
    .ns-n { background: #dc2626; }
    .ns-m { color: #71717a; }
    #ns-cite { background: #18181b; border-color: #27272a; }
    #ns-cite p { color: #a1a1aa !important; }
    #ns-cite code { background: #27272a !important; }
    #ns-cite strong { color: #d4d4d8 !important; }
    #ns-examples button {
        background: #18181b !important;
        border-color: #3f3f46 !important;
        color: #a1a1aa !important;
    }
    #ns-examples button:hover {
        border-color: #7f1d1d !important;
        background: #1c0a0a !important;
        color: #dc2626 !important;
    }
    .ns-ft { border-color: #27272a; }
    .ns-ft p { color: #52525b; }
    .ns-ft a { color: #dc2626; }
    /* Accordion dark */
    #ns-detail p, #ns-detail span, #ns-detail li, #ns-detail em { color: #d4d4d8 !important; }
    #ns-detail strong { color: #fafafa !important; }
    #ns-detail code { color: #a1a1aa !important; background: #27272a !important; }
    #ns-detail hr { border-color: #3f3f46 !important; }
    #ns-detail blockquote { background: #1c0a0a !important; border-color: #7f1d1d !important; color: #d4d4d8 !important; }
}

@media (max-width: 640px) {
    .ns-hero h1 { font-size: 1.25rem; }
    .ns-hero .ns-pill { font-size: 0.55rem; }
}
"""

# ── Formatters ────────────────────────────────────────────────────────────────

def _format_source_cards(results) -> str:
    if not results:
        return ""
    cards = []
    for i, r in enumerate(results, 1):
        speaker = r.speaker or "—"
        col = r.collection or ""
        date = r.date_iso or ""
        meta = " · ".join(p for p in [col, date] if p)
        cards.append(
            f'<span class="ns-c">'
            f'<span class="ns-n">{i}</span>'
            f'<span>{speaker}</span>'
            f'<span class="ns-m">{meta}</span>'
            f'</span>'
        )
    return f'<div class="ns-cards">{" ".join(cards)}</div>'


def _format_sources_detail(results) -> str:
    if not results:
        return "No sources retrieved."
    lines = []
    for i, r in enumerate(results, 1):
        rerank = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "n/a"
        lines.append(
            f"**[{i}]** `{r.collection}` · {r.date_iso or '?'} · "
            f"speaker: *{r.speaker or '—'}* · page {r.page_number or '?'} · "
            f"rerank: `{rerank}`\n\n"
            f"> {r.body[:400]}{'…' if len(r.body) > 400 else ''}"
        )
    return "\n\n---\n\n".join(lines)


def _format_citation_report(report: dict, cache_hit: bool = False) -> str:
    if not report:
        return ""
    ok = report.get("clean", False)
    status = "✓ All citations verified" if ok else "⚠ Issues detected"
    cache_label = "HIT" if cache_hit else "MISS"
    lines = [f"**{status}** · Groq `{GROQ_MODEL}` · Cache {cache_label}"]
    if report.get("cited"):
        lines.append(f"**Cited:** [{report['cited']}]")
    if report.get("hallucinated"):
        lines.append(f"**Hallucinated (stripped):** {report['hallucinated']}")
    if report.get("uncited_sources"):
        lines.append(f"**Unused sources:** [{report['uncited_sources']}]")
    if report.get("uncited_sentences"):
        lines.append(f"**Uncited claims:** {len(report['uncited_sentences'])}")
    stats = scholar.cache_stats
    lines.append(
        f"**Cache:** {stats['size']} entries · "
        f"{stats['hits']}/{stats['hits']+stats['misses']} hit rate "
        f"({stats['hit_rate']:.0%})"
    )
    return "  \n".join(lines)

# ── Query handler ─────────────────────────────────────────────────────────────

def gradio_query(query: str):
    if not query.strip():
        return "", "Please enter a question.", "", ""
    result = scholar.answer(query, top_k=5)
    sources = result["sources"]
    return (
        _format_source_cards(sources),
        result["answer"],
        _format_sources_detail(sources),
        _format_citation_report(result["citation_report"], result.get("cache_hit", False)),
    )

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Nuremberg Scholar", theme=theme, css=CUSTOM_CSS) as app:

    gr.HTML("""
    <div class="ns-hero">
        <h1>Nuremberg Scholar</h1>
        <p>Research assistant for the International Military Tribunal, 1945–1946</p>
        <div class="ns-pills">
            <span class="ns-pill">46,325 passages</span>
            <span class="ns-pill">BGE-M3 hybrid retrieval</span>
            <span class="ns-pill">Llama-3.1-8B via Groq</span>
        </div>
    </div>
    """)

    # Search bar — label is visible (fixes the invisible input bug)
    with gr.Row(elem_id="ns-search-row"):
        query_box = gr.Textbox(
            label="Question",
            placeholder="e.g. What did Speer claim about his knowledge of slave labour?",
            lines=1,
            max_lines=3,
            elem_id="ns-query",
            scale=6,
        )
        submit_btn = gr.Button(
            "Search",
            variant="primary",
            size="lg",
            elem_id="ns-search",
            scale=1,
        )

    # Examples — 4 compact chips
    gr.Examples(
        examples=[
            ["What did Goering say in his defense about the Luftwaffe?"],
            ["How did the Tribunal define crimes against humanity?"],
            ["What evidence was presented about the Final Solution?"],
            ["How were the defendants sentenced on 1 October 1946?"],
        ],
        inputs=[query_box],
        label="",
        examples_per_page=4,
        elem_id="ns-examples",
    )

    # Results
    source_cards = gr.HTML(value="", elem_id="ns-sources")
    answer_box = gr.Markdown(elem_id="ns-answer", show_label=False)
    citation_box = gr.Markdown(elem_id="ns-cite")

    with gr.Accordion("View full source passages", open=False, elem_id="ns-detail"):
        detail_box = gr.Markdown()

    gr.HTML("""
    <div class="ns-ft">
        <p>
            <a href="https://avalon.law.yale.edu/subject_menus/imt.asp" target="_blank">Yale Avalon Project</a> ·
            <a href="https://huggingface.co/datasets/dtufail/nuremberg-trials-corpus" target="_blank">dtufail/nuremberg-trials-corpus</a> ·
            CC BY 4.0
        </p>
    </div>
    """)

    outputs = [source_cards, answer_box, detail_box, citation_box]
    submit_btn.click(fn=gradio_query, inputs=[query_box], outputs=outputs)
    query_box.submit(fn=gradio_query, inputs=[query_box], outputs=outputs)

# ── Launch ────────────────────────────────────────────────────────────────────

app.launch(share=True)