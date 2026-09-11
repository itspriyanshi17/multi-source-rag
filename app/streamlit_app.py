"""Chat frontend for the NovaTech RAG API — supports both the fixed demo dataset and a
per-session file upload mode (CSV/PDF/DOCX/TXT/MD).

Run the API first: uvicorn app.api:app --reload
Then: streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NovaTech RAG Assistant", page_icon="🛠️")
st.title("🛠️ NovaTech RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # list of (filename, summary)

with st.sidebar:
    mode = st.radio("Data source", ["NovaTech demo data", "My uploaded files"])

    if mode == "NovaTech demo data":
        st.caption(
            "Routes between the product catalog (SQL) and policy/support documents "
            "(vector search + re-ranking) automatically."
        )
        st.subheader("Try asking:")
        for ex in [
            "How many laptops are under $1000 and in stock?",
            "What is the return window for accessories?",
            "Is the NovaView 27 QHD in stock, and is it covered under warranty for dead pixels?",
            "If I registered my NovaBook Pro 15 laptop 20 days ago, is a refund still possible?",
            "Who is the CEO of NovaTech Electronics?",
        ]:
            if st.button(ex, use_container_width=True):
                st.session_state.pending_question = ex
    else:
        st.caption("Upload a CSV, PDF, DOCX, TXT, or MD file, then ask questions about it.")
        uploaded = st.file_uploader(
            "Upload a file", type=["csv", "pdf", "docx", "txt", "md"], key="uploader"
        )
        if uploaded is not None and uploaded.name not in [f for f, _ in st.session_state.uploaded_files]:
            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    data = {"session_id": st.session_state.session_id} if st.session_state.session_id else {}
                    resp = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.session_id = result["session_id"]
                        st.session_state.uploaded_files.append((uploaded.name, result["summary"]))
                        st.success(result["summary"])
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.status_code)}")
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API server. Start it with: uvicorn app.api:app --reload")

        if st.session_state.uploaded_files:
            st.subheader("Uploaded this session:")
            for fname, summary in st.session_state.uploaded_files:
                st.markdown(f"- **{fname}** — {summary}")
            if st.button("Clear session / start over", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.uploaded_files = []
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("No files uploaded yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("context_used"):
            with st.expander("Sources / retrieved context"):
                st.text(msg["context_used"])

placeholder = (
    "Ask about products, policies, or support..."
    if mode == "NovaTech demo data"
    else "Ask about your uploaded files..."
)
question = st.chat_input(placeholder)
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    if mode == "My uploaded files" and not st.session_state.uploaded_files:
        st.warning("Upload a file first, in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                payload = {"question": question}
                if mode == "My uploaded files":
                    payload["session_id"] = st.session_state.session_id
                try:
                    resp = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        context_used = data["context_used"]
                    elif resp.status_code == 503:
                        answer = "⚠️ The assistant is temporarily over its request quota — please try again shortly."
                        context_used = ""
                    else:
                        answer = f"⚠️ Request failed ({resp.status_code}): {resp.json().get('detail', 'unknown error')}"
                        context_used = ""
                except requests.exceptions.ConnectionError:
                    answer = (
                        "⚠️ Can't reach the API server. Make sure it's running: "
                        "`uvicorn app.api:app --reload`"
                    )
                    context_used = ""

            st.markdown(answer)
            if context_used:
                with st.expander("Sources / retrieved context"):
                    st.text(context_used)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "context_used": context_used}
        )
