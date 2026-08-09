import sys
import os
from pathlib import Path
import json

# Ensure src directory is in sys.path when Streamlit runs directly
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import requests
import streamlit as st
from core.session_manager import session_manager
from ui.components.styles import inject_custom_css
from config import settings

# Page Configuration
st.set_page_config(
    page_title=settings.PROJECT_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default"
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

# ================= SIDEBAR CONFIGURATION & NAVIGATION =================
st.sidebar.image("https://img.icons8.com/isometric/96/brain-laptop.png", width=70)
st.sidebar.title("⚡ RAG Platform")

# 1. Left Panel Main Navigation
st.sidebar.markdown("### 📌 Navigation")
nav_page = st.sidebar.radio(
    "Select View",
    options=[
        "💬 Interactive RAG", 
        "📹 YouTube Q&A",
        "🔍 Vector Explorer", 
        "📁 Document Hub", 
        "⚙️ System Dashboard"
    ],
    key="navigation_selection",
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

default_api_url = f"http://{settings.HOST}:{settings.PORT}" if settings.HOST != "0.0.0.0" else f"http://localhost:{settings.PORT}"

# 2. Session Manager
st.sidebar.subheader("📂 Session Manager")

try:
    sess_resp = requests.get(f"{default_api_url}/sessions", timeout=2).json()
    all_sessions = sess_resp.get("sessions", [])
except Exception:
    all_sessions = session_manager.list_sessions()

session_map = {f"{s['name']} ({s['message_count']} msgs)": s["session_id"] for s in all_sessions}
session_ids = list(session_map.values())
session_labels = list(session_map.keys())

current_idx = 0
if st.session_state.active_session_id in session_ids:
    current_idx = session_ids.index(st.session_state.active_session_id)

selected_label = st.sidebar.selectbox(
    "Active Session:",
    options=session_labels,
    index=current_idx if current_idx < len(session_labels) else 0
)

selected_session_id = session_map.get(selected_label, "default")
st.session_state.active_session_id = selected_session_id

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("➕ New", use_container_width=True):
        try:
            new_s = requests.post(f"{default_api_url}/sessions").json()["session"]
            st.session_state.active_session_id = new_s["session_id"]
        except Exception:
            new_s = session_manager.create_session()
            st.session_state.active_session_id = new_s["session_id"]
        st.rerun()

with col_s2:
    if st.button("🗑️ Delete", use_container_width=True):
        try:
            requests.delete(f"{default_api_url}/sessions/{st.session_state.active_session_id}")
        except Exception:
            session_manager.delete_session(st.session_state.active_session_id)
        st.session_state.active_session_id = "default"
        st.rerun()

st.sidebar.caption(f"Session ID: `{st.session_state.active_session_id}`")
st.sidebar.markdown("---")

# 3. System & API Settings
st.sidebar.subheader("⚙️ Settings & Connection")

# Theme Mode Selector
theme_mode = st.sidebar.selectbox(
    "🎨 UI Theme Mode",
    options=["🌙 Dark Mode", "☀️ Light Mode"],
    index=0 if st.session_state.get("ui_theme", "dark") == "dark" else 1,
    key="theme_mode_selector"
)
selected_theme = "dark" if "Dark" in theme_mode else "light"
st.session_state.ui_theme = selected_theme

# Inject Custom Aesthetics CSS (Dark or Light)
inject_custom_css(selected_theme)

api_base_url = st.sidebar.text_input("FastAPI Server URL", value=default_api_url)
top_k_val = st.sidebar.slider("Retrieval Top-K Chunks", min_value=1, max_value=15, value=4)

# Chat Avatars Configuration
col_ic1, col_ic2 = st.sidebar.columns(2)
with col_ic1:
    user_avatar = st.text_input("User Icon", value=settings.USER_AVATAR, help="Emoji or image URL")
with col_ic2:
    ai_avatar = st.text_input("AI Icon", value=settings.AI_AVATAR, help="Emoji or image URL")



# Test connection to FastAPI Server
is_connected = False
try:
    r = requests.get(f"{api_base_url}/health", timeout=2)
    if r.status_code == 200:
        is_connected = True
        st.sidebar.markdown('<div class="pulse-online"><div class="pulse-dot"></div> REST API Connected</div>', unsafe_allow_html=True)
    else:
        st.sidebar.warning(f"⚠️ API status: {r.status_code}")
except Exception:
    st.sidebar.error("🔴 Backend Offline. Run `python app.py --backend`")


# Header Banner
status_badge = '<div class="pulse-online"><div class="pulse-dot"></div> Online</div>' if is_connected else '<span style="color:#f43f5e;font-weight:600;">Offline</span>'

st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="hero-title">⚡ {settings.PROJECT_NAME}</div>
            <div class="hero-subtitle">High-performance Retrieval-Augmented Generation powered by FAISS & Gemini LLMs</div>
        </div>
        <div>
            {status_badge}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper Function to Fetch Active History via REST API
def get_active_history():
    """Fetches conversation history for the active session via REST API or local session manager."""
    try:
        r = requests.get(f"{api_base_url}/sessions/{st.session_state.active_session_id}", timeout=3)
        if r.status_code == 200:
            return r.json().get("history", [])
    except Exception:
        pass
    return session_manager.get_history(st.session_state.active_session_id)

# Helper Function to Stream RAG Query Tokens
def stream_rag_tokens(user_query, top_k):
    """Streams real-time RAG response tokens from REST endpoint or fallback search engine generator."""
    payload = {
        "query": user_query,
        "top_k": top_k,
        "session_id": st.session_state.active_session_id
    }
    
    sources_holder = []
    
    try:
        response = requests.post(
            f"{api_base_url}/query/stream",
            json=payload,
            stream=True,
            timeout=45
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            data = json.loads(json_str)
                            if data["type"] == "sources":
                                sources_holder.extend(data.get("sources", []))
                            elif data["type"] == "token":
                                yield data.get("content", "")
                        except Exception:
                            pass
        else:
            raise Exception(f"API Error ({response.status_code}): {response.text}")
    except Exception:
        # Fallback direct local streaming if API server is offline
        from api.deps import get_rag_search
        rag = get_rag_search()
        history = session_manager.get_history(st.session_state.active_session_id)
        
        full_text = ""
        local_sources = []
        for event in rag.search_with_sources_stream(user_query, top_k=top_k, chat_history=history):
            if event["type"] == "sources":
                sources_holder.extend(event.get("sources", []))
                local_sources = event.get("sources", [])
            elif event["type"] == "token":
                token = event.get("content", "")
                full_text += token
                yield token
            elif event["type"] == "done":
                session_manager.add_message_pair(
                    session_id=st.session_state.active_session_id,
                    user_query=user_query,
                    assistant_summary=full_text,
                    sources=local_sources
                )
    
    st.session_state["latest_sources"] = sources_holder


# ================= MAIN CONTENT AREA (DYNAMIC VIEW) =================
if nav_page == "💬 Interactive RAG":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f'<div class="session-badge">⚡ Active Session: {st.session_state.active_session_id}</div>', unsafe_allow_html=True)
    with col_h2:
        if st.button("🗑️ Clear History", key="clear_chat_tab_btn", use_container_width=True):
            try:
                requests.delete(f"{api_base_url}/sessions/{st.session_state.active_session_id}")
            except Exception:
                session_manager.clear_session(st.session_state.active_session_id)
            st.rerun()

    # Quick Prompt Badges
    st.markdown("**💡 Quick Question Suggestions:**")
    cols_btn = st.columns(3)
    preset_query = None
    if cols_btn[0].button("🎓 Where did Shubham study?"):
        preset_query = "Where did Shubham study?"
    if cols_btn[1].button("💻 What are his key technical skills?"):
        preset_query = "What are his key skills?"
    if cols_btn[2].button("📄 Summarize all loaded documents"):
        preset_query = "Summarize the key information in all available documents."

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Active Session Chat History
    chat_history = get_active_history()
    for chat in chat_history:
        with st.chat_message("user", avatar=user_avatar):
            st.write(chat["query"])
        with st.chat_message("assistant", avatar=ai_avatar):
            st.markdown(chat["summary"])
            if chat.get("sources"):
                with st.expander(f"📚 Retrieved Context Sources ({len(chat['sources'])} chunks)"):
                    for idx, src in enumerate(chat["sources"]):
                        dist = src.get("distance", 0.0)
                        confidence = (1.0 / (1.0 + dist)) * 100.0
                        text = src.get("text", "")
                        meta = src.get("metadata", {})
                        source_file = meta.get('source', 'Document') if isinstance(meta, dict) else 'Document'
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <strong>Chunk #{idx+1} &bull; <code>{source_file}</code></strong>
                                <span class="score-meter">Relevance: {confidence:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.code(text, language="markdown")

    # Chat Input Box
    user_input = st.chat_input("Ask any question from your document knowledge base...")
    final_query = preset_query or user_input

    if final_query:
        with st.chat_message("user", avatar=user_avatar):
            st.write(final_query)
        
        with st.chat_message("assistant", avatar=ai_avatar):
            full_response = st.write_stream(stream_rag_tokens(final_query, top_k_val))
            
            latest_sources = st.session_state.get("latest_sources", [])
            if latest_sources:
                with st.expander(f"📚 Retrieved Context Sources ({len(latest_sources)} chunks)"):
                    for idx, src in enumerate(latest_sources):
                        dist = src.get("distance", 0.0)
                        confidence = (1.0 / (1.0 + dist)) * 100.0
                        text = src.get("text", "")
                        meta = src.get("metadata", {})
                        source_file = meta.get('source', 'Document') if isinstance(meta, dict) else 'Document'
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <strong>Chunk #{idx+1} &bull; <code>{source_file}</code></strong>
                                <span class="score-meter">Relevance: {confidence:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.code(text, language="markdown")
            st.rerun()


elif nav_page == "📹 YouTube Q&A":
    st.subheader("📹 YouTube Video Transcript & Q&A")
    st.caption("Extract YouTube transcripts with timestamps, view video content, and index into NexaMind vector store for AI Q&A.")

    yt_col1, yt_col2 = st.columns([3, 1])
    with yt_col1:
        yt_url = st.text_input("Enter YouTube Video URL or Video ID:", placeholder="https://www.youtube.com/watch?v=jNQXAC9IVRw", key="yt_url_input")
    with yt_col2:
        auto_index_yt = st.checkbox("Auto-index into Vector Store", value=True, key="yt_auto_index")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        fetch_btn = st.button("🚀 Fetch Transcript & Index", use_container_width=True, key="fetch_yt_btn")

    if fetch_btn and yt_url.strip():
        with st.spinner("Extracting transcript from YouTube..."):
            try:
                resp = requests.post(
                    f"{api_base_url}/youtube/transcript",
                    json={"url": yt_url.strip(), "save_to_dataset": True, "auto_reindex": auto_index_yt},
                    timeout=25
                )
                if resp.status_code == 200:
                    st.session_state["yt_data"] = resp.json()
                    st.success(f"Transcript fetched successfully! ({st.session_state['yt_data']['segment_count']} segments)")
                else:
                    st.error(f"Failed to fetch transcript via API: {resp.text}")
            except Exception:
                # Direct local fallback
                try:
                    from core.youtube_loader import fetch_youtube_transcript, save_transcript_to_dataset
                    yt_res = fetch_youtube_transcript(yt_url.strip())
                    saved_f = save_transcript_to_dataset(yt_res, settings.DATA_DIR)
                    if auto_index_yt:
                        from api.deps import get_rag_search
                        rag = get_rag_search()
                        indexed_c = rag.rebuild_index(settings.DATA_DIR)
                    else:
                        indexed_c = 0
                    yt_res["saved_file"] = saved_f.name
                    yt_res["indexed_documents_count"] = indexed_c
                    st.session_state["yt_data"] = yt_res
                    st.success(f"Transcript fetched successfully! ({yt_res['segment_count']} segments)")
                except Exception as ex:
                    st.error(f"Error fetching YouTube transcript: {str(ex)}")

    if "yt_data" in st.session_state and st.session_state["yt_data"]:
        yt_data = st.session_state["yt_data"]
        
        st.markdown("<hr style='border-color:rgba(255,255,255,0.08)'>", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{yt_data['video_id']}</div>
                <div class="glass-label">Video ID</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{yt_data['segment_count']}</div>
                <div class="glass-label">Transcript Segments</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            sf_name = yt_data.get('saved_file', 'Indexed')
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value" style="color:#34d399; font-size:1.1rem;">{sf_name}</div>
                <div class="glass-label">Dataset File Status</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        t_tab1, t_tab2, t_tab3 = st.tabs(["⏱️ Timestamped Transcript", "📝 Plain Text & Download", "⚡ Ask Video Questions"])
        
        with t_tab1:
            st.video(yt_data["url"])
            st.markdown("#### Timestamped Captions:")
            with st.container(height=350):
                for seg in yt_data.get("segments", []):
                    st.markdown(f"**`[{seg['timestamp']}]`** {seg['text']}")
                    
        with t_tab2:
            st.markdown("#### Complete Formatted Transcript:")
            st.text_area("Transcript Content:", value=yt_data.get("full_text", ""), height=300, key="yt_full_text_area")
            st.download_button(
                label="📥 Download Transcript (.txt)",
                data=yt_data.get("full_text", ""),
                file_name=f"transcript_{yt_data['video_id']}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        with t_tab3:
            st.markdown("#### 💬 Interactive Video Chat & Q&A")
            st.caption("Ask any question specifically about this video transcript. The AI will answer based on the video content.")

            if "yt_chat_history" not in st.session_state:
                st.session_state["yt_chat_history"] = []

            # Quick Suggestion Badges
            st.markdown("**💡 Quick Suggestions:**")
            p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 1])
            selected_preset = None
            if p_col1.button("🎯 Summarize Video", key="yt_btn_sum", use_container_width=True):
                selected_preset = f"Summarize the main key points of YouTube video {yt_data['video_id']}."
            if p_col2.button("🔑 Key Takeaways", key="yt_btn_takeaways", use_container_width=True):
                selected_preset = f"What are the top 3-5 key takeaways from video {yt_data['video_id']}?"
            if p_col3.button("🛠️ Methods & Steps", key="yt_btn_methods", use_container_width=True):
                selected_preset = f"What processes, tools, or steps were discussed in video {yt_data['video_id']}?"
            if p_col4.button("🗑️ Clear Chat", key="yt_btn_clear", use_container_width=True):
                st.session_state["yt_chat_history"] = []
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Display Video Chat History
            with st.container(height=380):
                if not st.session_state["yt_chat_history"]:
                    st.info("👋 Ask any question below or click a quick suggestion to start chatting about this video!")
                else:
                    for msg in st.session_state["yt_chat_history"]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                            if msg.get("sources"):
                                with st.expander(f"📚 Context Chunks Used ({len(msg['sources'])})"):
                                    for idx, src in enumerate(msg["sources"]):
                                        st.markdown(f"**Chunk #{idx+1}** | Distance: `{src.get('distance', 0.0):.4f}`")
                                        st.code(src.get("text", ""), language="markdown")

            # Chat Input Box
            yt_user_query = st.chat_input("Ask a question about this video...", key="yt_tab_chat_input")
            final_yt_query = selected_preset or yt_user_query

            if final_yt_query:
                st.session_state["yt_chat_history"].append({"role": "user", "content": final_yt_query})
                with st.chat_message("assistant"):
                    full_response = st.write_stream(stream_rag_tokens(final_yt_query, top_k=5))
                    latest_srcs = st.session_state.get("latest_sources", [])
                    st.session_state["yt_chat_history"].append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": latest_srcs
                    })
                st.rerun()


elif nav_page == "🔍 Vector Explorer":
    st.subheader("🔍 Vector Similarity Explorer")
    st.caption("Inspect raw FAISS vector distance scores and extracted text chunks without LLM summarization.")
    
    col_v1, col_v2 = st.columns([3, 1])
    with col_v1:
        vec_query = st.text_input("Enter search query for raw similarity match:", value="Shubham education details")
    with col_v2:
        vec_top_k = st.slider("Top K matches:", 1, 10, 5, key="vec_top_k")
    
    if st.button("🔍 Search FAISS Index", use_container_width=True):
        if vec_query.strip():
            with st.spinner("Executing FAISS vector search..."):
                try:
                    resp = requests.post(f"{api_base_url}/search", json={"query": vec_query, "top_k": vec_top_k})
                    results = resp.json().get("results", [])
                    
                    st.success(f"Retrieved {len(results)} vector matches!")
                    for idx, res in enumerate(results):
                        dist = res.get("distance", 0.0)
                        confidence = (1.0 / (1.0 + dist)) * 100.0
                        meta = res.get("metadata", {})
                        text = meta.get("text", "No text found") if meta else "N/A"
                        
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown(f"""
                            <div class="glass-card">
                                <div class="glass-value">#{idx+1}</div>
                                <div class="glass-label">Dist: {dist:.4f}</div>
                                <div style="margin-top:6px;"><span class="score-meter">{confidence:.1f}% match</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"**Chunk Index:** `{res.get('index', 'N/A')}`")
                            st.text_area("Extracted Context Text:", value=text, height=120, key=f"raw_text_{idx}")
                        st.markdown("<hr style='border-color:rgba(255,255,255,0.08)'>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Vector search failed: {str(e)}")

elif nav_page == "📁 Document Hub":
    st.subheader("📁 Document Knowledge Base Manager")
    st.caption("Upload documents to your knowledge dataset and rebuild vector indexes dynamically.")
    
    col_up, col_list = st.columns([1, 1])
    
    with col_up:
        st.markdown("#### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Supported file types: PDF, TXT, CSV, DOCX, XLSX, JSON",
            accept_multiple_files=True,
            type=["pdf", "txt", "csv", "docx", "xlsx", "json"]
        )
        auto_index_check = st.checkbox("Automatically rebuild vector index after upload", value=True)
        
        if st.button("🚀 Process & Index Files", use_container_width=True):
            if uploaded_files:
                with st.spinner("Uploading and indexing documents via FastAPI..."):
                    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    data_payload = {"auto_reindex": str(auto_index_check).lower()}
                    res = requests.post(f"{api_base_url}/upload", files=files_payload, data=data_payload)
                    if res.status_code == 200:
                        st.success(res.json().get("message", "Upload successful!"))
                    else:
                        st.error(f"Upload failed via API: {res.text}")
            else:
                st.warning("Please select at least one document to upload.")

    with col_list:
        st.markdown("#### 📂 Active Document Dataset")
        try:
            docs_res = requests.get(f"{api_base_url}/documents", timeout=2).json()
            doc_files = docs_res.get("documents", [])
            if doc_files:
                for idx, doc in enumerate(doc_files):
                    fname = doc["filename"]
                    fsize = doc["size_bytes"] / 1024.0
                    ext = doc.get("extension", "").upper()
                    
                    c_doc1, c_doc2 = st.columns([4, 1])
                    with c_doc1:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:8px 14px; border-radius:10px; display:flex; justify-content:space-between; align-items:center;">
                            <span>📄 <strong><code>{fname}</code></strong></span>
                            <span style="font-size:0.8rem; color:#9ca3af;">{fsize:.1f} KB &bull; <b style="color:#34d399">{ext}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_doc2:
                        if st.button("🗑️", key=f"del_doc_{idx}", help=f"Remove '{fname}' from dataset"):
                            try:
                                r_del = requests.delete(f"{api_base_url}/documents/{fname}")
                                if r_del.status_code == 200:
                                    st.toast(f"Deleted '{fname}'", icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {r_del.text}")
                            except Exception as ex:
                                target_path = settings.DATA_DIR / fname
                                if target_path.exists():
                                    target_path.unlink()
                                    from api.deps import get_rag_search
                                    get_rag_search().rebuild_index(settings.DATA_DIR)
                                    st.toast(f"Deleted '{fname}'", icon="🗑️")
                                    st.rerun()
            else:
                st.info("No documents uploaded yet.")
        except Exception:
            st.info("Unable to fetch document list from REST API.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔄 Rebuild Index", use_container_width=True):
                with st.spinner("Rebuilding FAISS vector index..."):
                    try:
                        r = requests.post(f"{api_base_url}/reindex")
                        st.success(r.json().get("message", "Reindexed successfully!"))
                    except Exception as e:
                        st.error(f"Reindex failed: {str(e)}")
        with col_b2:
            if st.button("🗑️ Clear All Data", use_container_width=True, help="Remove all documents and wipe vector store"):
                with st.spinner("Wiping dataset & clearing vector index..."):
                    try:
                        r_clr = requests.delete(f"{api_base_url}/documents")
                        if r_clr.status_code == 200:
                            st.success("All ingested data cleared successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to clear documents: {r_clr.text}")
                    except Exception as e:
                        st.error(f"Clear failed: {str(e)}")


elif nav_page == "⚙️ System Dashboard":
    st.subheader("⚙️ System Monitor & REST Endpoints")
    st.caption("Live operational state and REST API documentation.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        h_res = requests.get(f"{api_base_url}/health", timeout=3).json()
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{h_res.get('status', 'unknown').upper()}</div>
                <div class="glass-label">API Health</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{h_res.get('total_vectors', 0)}</div>
                <div class="glass-label">FAISS Vectors</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{h_res.get('active_sessions', 0)}</div>
                <div class="glass-label">Active Sessions</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-value">{h_res.get('data_files_count', 0)}</div>
                <div class="glass-label">Data Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌐 Registered REST API Endpoints")
        
        st.markdown("""
        | Method | Endpoint | Description |
        | :---: | :--- | :--- |
        | <span class="http-get">GET</span> | `/health` | Live system health metrics & vector count |
        | <span class="http-post">POST</span> | `/query` | Session-aware RAG search & Gemini AI generation |
        | <span class="http-post">POST</span> | `/query/stream` | Real-time SSE token streaming RAG endpoint |
        | <span class="http-post">POST</span> | `/search` | Raw FAISS vector similarity search |
        | <span class="http-get">GET</span> | `/sessions` | List active chat sessions |
        | <span class="http-post">POST</span> | `/sessions` | Create new chat session |
        | <span class="http-delete">DELETE</span> | `/sessions/{id}` | Clear turn history or delete session |
        | <span class="http-get">GET</span> | `/documents` | List uploaded dataset files |
        | <span class="http-post">POST</span> | `/upload` | Upload & auto-reindex documents |
        | <span class="http-post">POST</span> | `/reindex` | Force rebuild FAISS index |
        | <span class="http-post">POST</span> | `/youtube/transcript` | Extract YouTube transcript & optionally auto-index into FAISS vector store |
        | <span class="http-delete">DELETE</span> | `/documents/{filename}` | Delete single file & update vector index |
        | <span class="http-delete">DELETE</span> | `/documents` | Clear all dataset files & wipe vector store |
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"📖 **Interactive Swagger UI OpenAPI Documentation:** [{api_base_url}/docs]({api_base_url}/docs)")
        
    except Exception as ex:
        st.error(f"Unable to reach REST backend at {api_base_url}: {str(ex)}")
