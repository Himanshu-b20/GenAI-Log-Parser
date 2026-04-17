import streamlit as st
from rag import process_data, generate_answer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Log Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.85;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #1a1d2e 0%, #12151f 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-banner h1 {
    margin: 0;
    font-size: 1.5rem;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-banner p {
    margin: 0;
    font-size: 0.82rem;
    color: #6b7280;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 99px;
}
.status-ready   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-pending { background: #1c1917; color: #a8a29e; border: 1px solid #44403c; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.2rem 0.4rem;
}

/* ── Source chip ── */
.source-chip {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2e3250;
    color: #818cf8;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 99px;
    margin: 2px 3px 0 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    color: #4b5563;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.6rem; }
.empty-state h3   { color: #6b7280; font-weight: 500; margin-bottom: 0.3rem; }

/* ── Processing log ── */
.proc-step {
    font-size: 0.82rem;
    color: #94a3b8;
    padding: 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "db_ready"   not in st.session_state: st.session_state.db_ready   = False
if "file_names" not in st.session_state: st.session_state.file_names = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📥 Logs Ingestion")
    st.markdown("---")

    # Status badge
    if st.session_state.db_ready:
        st.markdown(
            '<span class="status-badge status-ready">● Vector DB ready</span>',
            unsafe_allow_html=True,
        )
        if st.session_state.file_names:
            st.markdown(
                "<p style='font-size:0.78rem;color:#6b7280;margin-top:6px'>"
                + "  \n".join(f"📄 {n}" for n in st.session_state.file_names)
                + "</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="status-badge status-pending">○ No logs loaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown("### Upload Log Files")
    uploaded_files = st.file_uploader(
        "Drop `.txt` log files here",
        type=["txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_clicked = st.button("⚡ Process Logs", disabled=not uploaded_files)

    if st.session_state.db_ready:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.72rem;color:#374151'>"
        "Powered by LangChain · Groq · ChromaDB</p>",
        unsafe_allow_html=True,
    )

# ── Process uploaded files ─────────────────────────────────────────────────────
if process_clicked and uploaded_files:
    combined_text = ""
    for file in uploaded_files:
        combined_text += file.read().decode("utf-8") + "\n"

    with open("temp_file.txt", "w") as f:
        f.write(combined_text)

    with st.sidebar:
        proc_box = st.empty()
        steps = []
        for status in process_data("temp_file.txt"):
            steps.append(status)
            proc_box.markdown(
                "".join(f'<div class="proc-step">✓ {s}</div>' for s in steps),
                unsafe_allow_html=True,
            )

    st.session_state.db_ready   = True
    st.session_state.file_names = [f.name for f in uploaded_files]

    # System message in chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            f"✅ Loaded **{len(uploaded_files)} file(s)** into the vector database — "
            f"{', '.join(f'`{f.name}`' for f in uploaded_files)}.\n\n"
            "Ask me anything about your logs!"
        ),
        "sources": "",
    })
    st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div style="font-size:2rem">🔍</div>
  <div>
    <h1>Log Analyser</h1>
    <p>Chat with your application logs — errors, traces, anomalies & more</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📂</div>
          <h3>No logs loaded yet</h3>
          <p>Upload <code>.txt</code> log files in the sidebar and click <strong>Process Logs</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    src_list = [s.strip() for s in msg["sources"].split(",") if s.strip()]
                    if src_list:
                        chips = "".join(
                            f'<span class="source-chip">📎 {s}</span>' for s in src_list
                        )
                        st.markdown(
                            f'<div style="margin-top:6px">{chips}</div>',
                            unsafe_allow_html=True,
                        )

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask about your logs…" if st.session_state.db_ready else "Upload and process logs first…",
    disabled=not st.session_state.db_ready,
)

if prompt:
    # 1. Persist & render the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": ""})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Open the assistant bubble straight away so it appears without delay
    with st.chat_message("assistant", avatar="🤖"):
        # Typing indicator lives inside the bubble — no full-page grey overlay
        answer_placeholder  = st.empty()
        sources_placeholder = st.empty()

        answer_placeholder.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;color:#818cf8;font-size:0.9rem">
              <span>Analysing logs</span>
              <span style="display:inline-flex;gap:4px">
                <span style="animation:blink 1.2s infinite 0.0s;opacity:0">●</span>
                <span style="animation:blink 1.2s infinite 0.4s;opacity:0">●</span>
                <span style="animation:blink 1.2s infinite 0.8s;opacity:0">●</span>
              </span>
            </div>
            <style>
              @keyframes blink {
                0%%,80%%,100%% { opacity:0 }
                40%%           { opacity:1 }
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 3. Run the (blocking) LLM call — only this bubble shows the indicator
        try:
            answer, sources = generate_answer(prompt)
        except RuntimeError as e:
            answer  = f"⚠️ Error: {e}"
            sources = ""

        # 4. Replace indicator with the real answer
        answer_placeholder.markdown(answer)

        if sources:
            src_list = [s.strip() for s in sources.split(",") if s.strip()]
            if src_list:
                chips = "".join(
                    f'<span class="source-chip">📎 {s}</span>' for s in src_list
                )
                sources_placeholder.markdown(
                    f'<div style="margin-top:6px">{chips}</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
