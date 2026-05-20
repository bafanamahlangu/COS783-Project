"""
app.py — Streamlit web front-end for the Digital Forensics AI search tool.
COS783 Final Assignment 2026 | Option 1: Keyword Searching

Pipeline (mirrors main.py exactly):
  FileReader      → sentence-level chunking (NLTK punkt)
  SentenceTransformer → encode sentences once, cache embeddings
  cosine similarity   → rank sentences by semantic relevance to query
  extract_entities    → spaCy NER on each matched sentence

Key fixes vs previous version:
  - nltk punkt downloaded before FileReader is ever called (fixes full-email display)
  - Corpus embeddings pre-computed once and cached (fixes score discrepancy with CLI)
  - Single FileReader call per file (no duplicate reads)
  - Model loaded via @st.cache_resource so it survives Streamlit re-runs
"""

import io
import os
import tempfile
from pathlib import Path

import nltk
# ── Download punkt FIRST, before any FileReader call ─────────────────────────
# Without this, sent_tokenize silently returns the whole message as one
# "sentence", which causes (a) full emails to display and (b) much lower
# cosine scores than the CLI (which has punkt available from the old app).
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ── Project modules ───────────────────────────────────────────────────────────
from file_reader import FileReader
from keyword_searcher import extract_entities   # NER only; search done here directly

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForensicSearch · AI Keyword Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; background-color: #0a0e14; color: #c9d1d9; }
.stApp {
    background: #0a0e14;
    background-image: radial-gradient(ellipse at 10% 0%, rgba(0,255,136,0.04) 0%, transparent 55%),
                      radial-gradient(ellipse at 90% 100%, rgba(0,180,255,0.04) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1c2a1e !important; }
[data-testid="stFileUploader"] { background: #0d1117; border: 1px dashed rgba(46,160,67,0.25) !important; border-radius: 8px; }
.stTextInput > div > div > input {
    background: #0d1117 !important; border: 1px solid #21262d !important; border-radius: 6px !important;
    color: #e6edf3 !important; font-family: 'Share Tech Mono', monospace !important;
}
.stTextInput > div > div > input:focus { border-color: #2ea043 !important; box-shadow: 0 0 0 3px rgba(46,160,67,0.15) !important; }
.stButton > button {
    background: #2ea043 !important; color: #ffffff !important; border: none !important;
    border-radius: 6px !important; font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #3fb950 !important; box-shadow: 0 0 20px rgba(46,160,67,0.35) !important; transform: translateY(-1px) !important; }
[data-testid="metric-container"] { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 0.8rem 1.2rem; }
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #2ea043 !important; font-family: 'Share Tech Mono', monospace !important; }
hr { border-color: #21262d !important; margin: 1.2rem 0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

.result-card { background: #0d1117; border: 1px solid #21262d; border-left: 3px solid #2ea043; border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 0.9rem; transition: border-color 0.2s; }
.result-card:hover { border-left-color: #3fb950; border-color: #30363d; }
.result-card.rank-1 { border-left-color: #ffa657; }
.result-rank { font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.3rem; }
.result-score { font-family: 'Share Tech Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #2ea043; }
.result-score.rank-1 { color: #ffa657; }
.result-id { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: #8b949e; }
.result-text { font-size: 1rem; color: #e6edf3; line-height: 1.6; margin: 0.5rem 0; }
.entity-tag { display: inline-block; font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 4px; margin: 0.15rem 0.2rem 0.15rem 0; font-weight: 600; letter-spacing: 0.04em; }
.tag-PERSON   { background: rgba(121,192,255,0.12); color: #79c0ff; border: 1px solid rgba(121,192,255,0.25); }
.tag-ORG      { background: rgba(210,153,34,0.12);  color: #d2a21f; border: 1px solid rgba(210,153,34,0.25); }
.tag-GPE, .tag-LOC { background: rgba(163,113,247,0.12); color: #a371f7; border: 1px solid rgba(163,113,247,0.25); }
.tag-DATE, .tag-TIME { background: rgba(87,171,90,0.12); color: #57ab5a; border: 1px solid rgba(87,171,90,0.25); }
.tag-MONEY    { background: rgba(255,123,114,0.12); color: #ff7b72; border: 1px solid rgba(255,123,114,0.25); }
.tag-default  { background: rgba(139,148,158,0.12); color: #8b949e; border: 1px solid rgba(139,148,158,0.25); }
.score-bar-bg  { width: 110px; height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 2px; }
.entity-divider { border-top: 1px solid #21262d; padding-top: 0.5rem; margin-top: 0.45rem; }
.entity-section-label { font-size: 0.65rem; color: #484f58; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; margin-right: 0.4rem; }
.info-box { background: rgba(121,192,255,0.05); border: 1px solid rgba(121,192,255,0.15); border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.85rem; color: #79c0ff; margin-bottom: 1rem; }
.warn-box { background: rgba(255,166,87,0.05); border: 1px solid rgba(255,166,87,0.2); border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.85rem; color: #ffa657; }
.section-label { font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# @st.cache_resource  → survives every Streamlit re-run (models, NLP pipelines)
# @st.cache_data      → re-runs when inputs change (file contents, queries)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load the same model used in keyword_searcher.py, cached for the session.

    Using cache_resource (not cache_data) ensures the model stays in memory
    across re-runs rather than being reloaded on every Streamlit interaction.
    This matches how keyword_searcher.py loads _MODEL at module level.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_sentences(file_path: str) -> tuple[list, list]:
    """Call FileReader once and cache sentences by file path.

    FileReader uses NLTK sent_tokenize internally. punkt must be downloaded
    BEFORE this is called — handled at the top of this file.
    Returns (unique_ids, sentences) as flat parallel lists.
    """
    reader = FileReader(file_path)
    unique_ids, sentences = reader.read_file()
    return unique_ids, sentences


@st.cache_data(show_spinner=False)
def compute_embeddings(sentences: tuple, _model) -> torch.Tensor:
    """Encode all sentences into vectors once and cache the result.

    Accepts sentences as a tuple (hashable) so Streamlit can use it as a
    cache key. Re-runs only when the sentence list actually changes.
    convert_to_tensor=True uses GPU if available, CPU otherwise.
    """
    return _model.encode(list(sentences), convert_to_tensor=True, show_progress_bar=False)


def semantic_search(
    query: str,
    unique_ids: list,
    sentences: list,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Encode query → cosine similarity → ranked results.

    This replicates KeywordSearcher.search_keywords() but operates on
    pre-computed embeddings so only the query needs encoding per search.
    Scores produced here are identical to what the CLI produces.
    """
    import re
    clean_query = re.sub(r"\s+", " ", query).strip()   # same normalise() as keyword_searcher.py
    query_emb   = model.encode(clean_query, convert_to_tensor=True)
    scores      = util.cos_sim(query_emb, embeddings)[0]
    top_indices = torch.topk(scores, k=min(top_k, len(sentences))).indices.tolist()

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        results.append({
            "rank":       len(results) + 1,
            "score":      score,
            "message_id": unique_ids[idx],
            "text":       sentences[idx],        # sentence, not full email
        })
    return results


def resolve_file_path(selected_option: str, uploaded_file) -> str | None:
    """Return a real filesystem path for FileReader.

    FileReader expects a path string. For uploaded files Streamlit provides
    bytes, so we write them to a named temp file and return its path.
    The temp file is kept in session_state to avoid being garbage-collected.
    """
    if selected_option != "⬆  Upload your own file…":
        return str(Path("Data") / selected_option)

    if uploaded_file is None:
        return None

    # Reuse existing temp file if the upload hasn't changed
    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_upload_key") == upload_key:
        return st.session_state["_tmp_path"]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="forensic_")
    tmp.write(uploaded_file.getvalue())
    tmp.close()

    # Clean up previous temp file
    old = st.session_state.get("_tmp_path")
    if old and os.path.exists(old):
        os.unlink(old)

    st.session_state["_upload_key"] = upload_key
    st.session_state["_tmp_path"]   = tmp.name
    return tmp.name


def score_color(score: float) -> str:
    if score >= 0.55: return "#2ea043"
    if score >= 0.35: return "#ffa657"
    return "#8b949e"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.1rem;
                    font-weight:700;color:#e6edf3;letter-spacing:0.06em;'>
            ⬡ FORENSIC<span style='color:#2ea043;'>SEARCH</span>
        </div>
        <div style='font-size:0.7rem;color:#484f58;font-family:"Share Tech Mono",monospace;
                    letter-spacing:0.08em;'>COS783 · AI FORENSICS PROJECT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📂 Data Source</div>', unsafe_allow_html=True)
    data_folder  = Path("Data")
    csv_files    = sorted(data_folder.glob("*.csv")) if data_folder.exists() else []
    file_options = ["⬆  Upload your own file…"] + [f.name for f in csv_files]
    selected_option = st.selectbox("Choose file", file_options, label_visibility="collapsed")

    uploaded_file = None
    if selected_option == "⬆  Upload your own file…":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">⚙ Search Settings</div>', unsafe_allow_html=True)
    top_k     = st.slider("Results to return",        1, 20, 5)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.25, 0.01)

    st.markdown("---")
    if st.button("🗑  Clear Cache & Restart", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""
    <div style='font-size:0.76rem;color:#484f58;line-height:1.7;margin-top:0.5rem;'>
        <b style='color:#8b949e;'>Pipeline</b><br>
        <code style='color:#c9d1d9;'>FileReader</code> splits messages into sentences.<br>
        <code style='color:#c9d1d9;'>SentenceTransformer</code> encodes sentences once,
        then scores each against your query using cosine similarity.<br>
        <code style='color:#c9d1d9;'>extract_entities</code> runs spaCy NER on each result.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:0.5rem 0 1.4rem 0;'>
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.4rem;
                font-weight:700;color:#e6edf3;letter-spacing:0.02em;line-height:1.15;'>
        Digital <span style='color:#2ea043;'>Forensics</span> AI
    </div>
    <div style='font-size:0.9rem;color:#8b949e;font-weight:300;margin-top:0.2rem;'>
        Semantic search across forensic text data — emails, messages, logs.
        Finds <em>meaning</em>, not just exact words.
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOADING
# ═══════════════════════════════════════════════════════════════════════════════
file_path = resolve_file_path(selected_option, uploaded_file)

if file_path is None:
    st.markdown("""
    <div class='info-box'>
        ℹ  Select a CSV from the sidebar or upload your own to get started.<br>
        Your file needs exactly <b>2 columns</b>: a unique ID and a message column.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Expected CSV Format</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "message_id": [1, 2, 3],
        "message":    [
            "We are live bro. Send the money for the drugs.",
            "Please wire the cash to Jamie.",
            "Meeting confirmed for Tuesday at the usual spot near downtown.",
        ]
    }), use_container_width=True, hide_index=True)
    st.stop()

# Load sentences via FileReader (NLTK punkt already downloaded above)
with st.spinner("Reading file…"):
    try:
        unique_ids, sentences = load_sentences(file_path)
    except Exception as e:
        st.error(f"FileReader error: {e}")
        st.stop()

if not sentences:
    st.markdown("""
    <div class='warn-box'>⚠ No sentences found. Check the file has exactly 2 columns (ID, message).</div>
    """, unsafe_allow_html=True)
    st.stop()

# Load model and pre-compute corpus embeddings
with st.spinner("Loading AI model…"):
    model = load_model()

with st.spinner(f"Computing embeddings for {len(sentences):,} sentences…"):
    embeddings = compute_embeddings(tuple(sentences), model)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
df_preview = pd.read_csv(file_path)
avg_len    = int(df_preview[df_preview.columns[1]].astype(str).apply(len).mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Messages",  f"{len(df_preview):,}")
m2.metric("Total Sentences", f"{len(sentences):,}")
m3.metric("ID Column",       df_preview.columns[0])
m4.metric("Avg Msg Length",  f"{avg_len} chars")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df_preview.head(10), use_container_width=True, hide_index=True)

st.markdown(f"""
<div style='font-family:"Share Tech Mono",monospace;font-size:0.78rem;color:#2ea043;margin-bottom:1rem;'>
    ● Model ready · {len(sentences):,} sentences indexed
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
col_q, col_btn = st.columns([5, 1.2])
with col_q:
    st.markdown('<div class="section-label">🔍 Search Query</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Query",
        placeholder="e.g.  drug transaction  ·  money laundering  ·  wire transfer",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown('<div style="height:1.55rem"></div>', unsafe_allow_html=True)
    search_clicked = st.button("SEARCH", use_container_width=True)

if search_clicked:
    if not query.strip():
        st.markdown("<div class='warn-box'>⚠ Please enter a search query.</div>",
                    unsafe_allow_html=True)
        st.stop()

    with st.spinner("Searching…"):
        results = semantic_search(
            query, unique_ids, sentences, embeddings, model, top_k, min_score
        )

    st.markdown("---")
    st.markdown(f"""
    <div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:1rem;'>
        <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.3rem;
                    font-weight:700;color:#e6edf3;letter-spacing:0.04em;'>RESULTS</div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.8rem;color:#8b949e;'>
            {len(results)} match{'es' if len(results) != 1 else ''} for
            <span style='color:#2ea043;'>"{query}"</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not results:
        st.markdown("""
        <div class='warn-box'>
            ⚠ No results above the minimum score. Lower the threshold or rephrase your query.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Result cards ──────────────────────────────────────────────────────────
    export_rows = []
    for r in results:
        # extract_entities imported from keyword_searcher.py
        entities = extract_entities(r["text"])
        rank_cls = "rank-1" if r["rank"] == 1 else ""
        bar_pct  = max(4, int(r["score"] * 100))
        bar_col  = score_color(r["score"])

        entity_html = ""
        for label, values in entities.items():
            for val in values:
                css = f"tag-{label}" if label in (
                    "PERSON","ORG","GPE","LOC","DATE","TIME","MONEY"
                ) else "tag-default"
                entity_html += f'<span class="entity-tag {css}">{label}: {val}</span>'

        entity_section = ""
        if entity_html:
            entity_section = f"""
            <div class="entity-divider">
                <span class="entity-section-label">Entities</span>
                {entity_html}
            </div>"""

        st.markdown(f"""
        <div class="result-card {rank_cls}">
            <div style='display:flex;align-items:center;justify-content:space-between;'>
                <div>
                    <div class="result-rank">Match #{r['rank']}</div>
                    <div style='display:flex;align-items:baseline;gap:6px;'>
                        <span class="result-score {rank_cls}">{r['score']:.4f}</span>
                        <span style='font-size:0.68rem;color:#484f58;
                                     font-family:"Share Tech Mono",monospace;'>similarity</span>
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div class="result-id">ID: {r['message_id']}</div>
                    <div class="score-bar-bg" style='margin-top:5px;margin-left:auto;'>
                        <div class="score-bar-fill"
                             style='width:{bar_pct}%;background:{bar_col};'></div>
                    </div>
                </div>
            </div>
            <div class="result-text">{r['text']}</div>
            {entity_section}
        </div>
        """, unsafe_allow_html=True)

        export_rows.append({
            "rank":       r["rank"],
            "score":      round(r["score"], 6),
            "message_id": r["message_id"],
            "text":       r["text"],
            "entities":   str(entities) if entities else "",
        })

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Export Results</div>', unsafe_allow_html=True)
    buf = io.StringIO()
    pd.DataFrame(export_rows).to_csv(buf, index=False)
    st.download_button(
        label="⬇  Download Results CSV",
        data=buf.getvalue(),
        file_name=f"forensic_results_{query[:30].replace(' ','_')}.csv",
        mime="text/csv",
    )