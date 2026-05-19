"""
app.py — Streamlit web front-end for the Digital Forensics AI search tool.

Integrates the existing project modules:
  - file_reader.FileReader       → sentence chunking
  - keyword_searcher.search_keywords / extract_entities → search + NER

Uploaded files are written to a temporary path so FileReader can accept them
with its normal file-path interface.
"""

import io
import tempfile
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import torch

# ── Project modules ───────────────────────────────────────────────────────────
# Import lazily inside @st.cache_resource so the heavy model load only happens
# once per Streamlit server session, not on every script re-run.
from file_reader import FileReader
from keyword_searcher import KeywordSearcher, extract_entities

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="ForensicSearch · AI Keyword Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0e14;
    color: #c9d1d9;
}
.stApp {
    background: #0a0e14;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(0,255,136,0.04) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 100%, rgba(0,180,255,0.04) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1c2a1e !important;
}
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 1px dashed rgba(46,160,67,0.25) !important;
    border-radius: 8px;
}
.stTextInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2ea043 !important;
    box-shadow: 0 0 0 3px rgba(46,160,67,0.15) !important;
}
.stButton > button {
    background: #2ea043 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #3fb950 !important;
    box-shadow: 0 0 20px rgba(46,160,67,0.35) !important;
    transform: translateY(-1px) !important;
}
[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.8rem 1.2rem;
}
[data-testid="metric-container"] label {
    color: #8b949e !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2ea043 !important; font-family: 'Share Tech Mono', monospace !important;
}
hr { border-color: #21262d !important; margin: 1.2rem 0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

.result-card {
    background: #0d1117; border: 1px solid #21262d;
    border-left: 3px solid #2ea043; border-radius: 8px;
    padding: 1.1rem 1.4rem; margin-bottom: 0.9rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-left-color: #3fb950; border-color: #30363d; }
.result-card.rank-1 { border-left-color: #ffa657; }
.result-rank {
    font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem;
    font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8b949e; margin-bottom: 0.3rem;
}
.result-score { font-family: 'Share Tech Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #2ea043; }
.result-score.rank-1 { color: #ffa657; }
.result-id { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: #8b949e; }
.result-text { font-size: 1rem; color: #e6edf3; line-height: 1.6; margin: 0.5rem 0; }
.entity-tag {
    display: inline-block; font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 4px;
    margin: 0.15rem 0.2rem 0.15rem 0; font-weight: 600; letter-spacing: 0.04em;
}
.tag-PERSON   { background: rgba(121,192,255,0.12); color: #79c0ff; border: 1px solid rgba(121,192,255,0.25); }
.tag-ORG      { background: rgba(210,153,34,0.12);  color: #d2a21f; border: 1px solid rgba(210,153,34,0.25); }
.tag-GPE, .tag-LOC { background: rgba(163,113,247,0.12); color: #a371f7; border: 1px solid rgba(163,113,247,0.25); }
.tag-DATE, .tag-TIME { background: rgba(87,171,90,0.12);  color: #57ab5a; border: 1px solid rgba(87,171,90,0.25); }
.tag-MONEY    { background: rgba(255,123,114,0.12); color: #ff7b72; border: 1px solid rgba(255,123,114,0.25); }
.tag-default  { background: rgba(139,148,158,0.12); color: #8b949e; border: 1px solid rgba(139,148,158,0.25); }
.score-bar-bg  { width: 110px; height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 2px; }
.entity-divider { border-top: 1px solid #21262d; padding-top: 0.5rem; margin-top: 0.45rem; }
.entity-section-label {
    font-size: 0.65rem; color: #484f58; letter-spacing: 0.08em;
    text-transform: uppercase; font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700; margin-right: 0.4rem;
}
.info-box {
    background: rgba(121,192,255,0.05); border: 1px solid rgba(121,192,255,0.15);
    border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.85rem;
    color: #79c0ff; margin-bottom: 1rem;
}
.warn-box {
    background: rgba(255,166,87,0.05); border: 1px solid rgba(255,166,87,0.2);
    border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.85rem; color: #ffa657;
}
.section-label {
    font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_file_path(selected_option: str, uploaded_file) -> str | None:
    """Return a real filesystem path that FileReader can consume.

    For files in the local Data/ folder this is trivial.
    For Streamlit uploads we write the bytes to a named temp file so
    FileReader's pd.read_csv(file_path) call works unchanged.

    The temp file is stored in st.session_state so it is not garbage-collected
    mid-session (NamedTemporaryFile with delete=True would vanish immediately).
    """
    if selected_option != "⬆  Upload your own file…":
        return str(Path("Data") / selected_option)

    if uploaded_file is None:
        return None

    # Re-use the same temp file if the uploaded file hasn't changed
    upload_key = f"_tmp_{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_upload_key") == upload_key:
        return st.session_state["_tmp_path"]

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="forensic_upload_"
    )
    tmp.write(uploaded_file.getvalue())
    tmp.close()

    # Clean up any previous temp file
    old_path = st.session_state.get("_tmp_path")
    if old_path and os.path.exists(old_path):
        os.unlink(old_path)

    st.session_state["_upload_key"] = upload_key
    st.session_state["_tmp_path"]   = tmp.name
    return tmp.name


@st.cache_data(show_spinner=False)
def load_records(file_path: str) -> tuple[list, list]:
    """Call FileReader and cache the result by file path.

    FileReader already splits each message into sentences via sent_tokenize,
    so no additional splitting is needed here.
    """
    reader = FileReader(file_path)
    unique_ids, chunks = reader.read_file()
    return unique_ids, chunks


@st.cache_data(show_spinner=False)
def run_search(keywords: str, file_path: str, top_k: int) -> tuple[list, list]:
    """Thin wrapper around KeywordSearcher so results are cached per query."""
    return KeywordSearcher.search_keywords(keywords, file_path, top_k=top_k)


def score_color(score: float) -> str:
    if score >= 0.55: return "#2ea043"
    if score >= 0.35: return "#ffa657"
    return "#8b949e"


# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    # ── File selection ──
    st.markdown('<div class="section-label">📂 Data Source</div>', unsafe_allow_html=True)

    data_folder  = Path("Data")
    csv_files    = sorted(data_folder.glob("*.csv")) if data_folder.exists() else []
    file_options = ["⬆  Upload your own file…"] + [f.name for f in csv_files]

    selected_option = st.selectbox("Choose file", file_options, label_visibility="collapsed")

    uploaded_file = None
    if selected_option == "⬆  Upload your own file…":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
                                          label_visibility="collapsed")

    # ── Settings ──
    st.markdown("---")
    st.markdown('<div class="section-label">⚙ Search Settings</div>', unsafe_allow_html=True)

    top_k     = st.slider("Results to return",        1, 20, 5)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.25, 0.01)

    st.markdown("---")
    if st.button("🗑  Clear Cache & Restart", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='font-size:0.76rem;color:#484f58;line-height:1.7;margin-top:0.5rem;'>
        <b style='color:#8b949e;'>Pipeline</b><br>
        <code style='color:#c9d1d9;'>FileReader</code> splits messages into sentences via NLTK.<br>
        <code style='color:#c9d1d9;'>KeywordSearcher</code> embeds with
        <b style='color:#c9d1d9;'>all-MiniLM-L6-v2</b> and ranks by cosine similarity.<br>
        <code style='color:#c9d1d9;'>extract_entities</code> surfaces names, dates, and
        money via spaCy NER.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
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

# ── Resolve file path ─────────────────────────────────────────────────────────
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
        "message": [
            "We are live bro. Send the money for the drugs.",
            "Please wire the cash to Jamie.",
            "Meeting confirmed for Tuesday at the usual spot near downtown.",
        ]
    }), use_container_width=True, hide_index=True)
    st.stop()

# ── Load and index records via FileReader ─────────────────────────────────────
with st.spinner("Reading and indexing file…"):
    try:
        unique_ids, chunks = load_records(file_path)
    except Exception as e:
        st.error(f"FileReader error: {e}")
        st.stop()

if not chunks:
    st.markdown("""
    <div class='warn-box'>
        ⚠ No sentences could be extracted from the file.
        Check that it has exactly 2 columns (ID, message).
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Dataset overview ──────────────────────────────────────────────────────────
df_preview = pd.read_csv(file_path)
avg_len    = int(df_preview[df_preview.columns[1]].astype(str).apply(len).mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Messages", f"{len(df_preview):,}")
m2.metric("Total Sentences", f"{len(chunks):,}")
m3.metric("ID Column",       df_preview.columns[0])
m4.metric("Avg Msg Length",  f"{avg_len} chars")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df_preview.head(10), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Search input ──────────────────────────────────────────────────────────────
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

# ── Run search via KeywordSearcher ────────────────────────────────────────────
if search_clicked:
    if not query.strip():
        st.markdown("<div class='warn-box'>⚠ Please enter a search query.</div>",
                    unsafe_allow_html=True)
        st.stop()

    with st.spinner("Searching…"):
        top_indices, top_scores = run_search(query, file_path, top_k)

    # Apply minimum score filter (KeywordSearcher returns top_k without a threshold)
    paired   = [(i, s) for i, s in zip(top_indices, top_scores) if s >= min_score]

    st.markdown("---")
    st.markdown(f"""
    <div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:1rem;'>
        <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.3rem;
                    font-weight:700;color:#e6edf3;letter-spacing:0.04em;'>RESULTS</div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.8rem;color:#8b949e;'>
            {len(paired)} match{'es' if len(paired) != 1 else ''} for
            <span style='color:#2ea043;'>"{query}"</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not paired:
        st.markdown("""
        <div class='warn-box'>
            ⚠ No results met the minimum similarity threshold.
            Try lowering the score filter or rephrasing your query.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    export_rows = []
    for rank, (idx, score) in enumerate(paired, start=1):
        chunk    = chunks[idx]
        msg_id   = unique_ids[idx]
        entities = extract_entities(chunk)          # from keyword_searcher.py
        rank_cls = "rank-1" if rank == 1 else ""
        bar_pct  = max(4, int(score * 100))
        bar_col  = score_color(score)

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
                    <div class="result-rank">Match #{rank}</div>
                    <div style='display:flex;align-items:baseline;gap:6px;'>
                        <span class="result-score {rank_cls}">{score:.4f}</span>
                        <span style='font-size:0.68rem;color:#484f58;
                                     font-family:"Share Tech Mono",monospace;'>similarity</span>
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div class="result-id">ID: {msg_id}</div>
                    <div class="score-bar-bg" style='margin-top:5px;margin-left:auto;'>
                        <div class="score-bar-fill"
                             style='width:{bar_pct}%;background:{bar_col};'></div>
                    </div>
                </div>
            </div>
            <div class="result-text">{chunk}</div>
            {entity_section}
        </div>
        """, unsafe_allow_html=True)

        export_rows.append({
            "rank":       rank,
            "score":      round(score, 6),
            "message_id": msg_id,
            "text":       chunk,
            "entities":   str(entities) if entities else "",
        })

    # ── Export ──
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