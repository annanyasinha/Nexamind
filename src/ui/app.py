import json
import os
import sys
from pathlib import Path
from textwrap import dedent

# Ensure src directory is in sys.path when Streamlit runs directly
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import requests
import streamlit as st

from config import settings
from ui.components.styles import inject_custom_css


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=settings.PROJECT_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default"

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

if "github_repo_data" not in st.session_state:
    st.session_state.github_repo_data = None

if "active_github_repo" not in st.session_state:
    st.session_state.active_github_repo = None

if "github_last_answer" not in st.session_state:
    st.session_state.github_last_answer = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div style="font-size:3.2rem; margin-bottom:-10px; margin-top:-15px;">🧠</div>',
    unsafe_allow_html=True,
)
st.sidebar.title("NexaMind RAG")

st.sidebar.markdown("### Navigation")

nav_page = st.sidebar.radio(
    "Select View",
    options=[
        "🤖 NexaMind AI Agent",
        "💬 Interactive RAG",
        "📹 YouTube Q&A",
        "💻 GitHub Q&A",
        "🔍 Vector Explorer",
        "📁 Document Hub",
        "⚙️ System Dashboard",
    ],
    key="navigation_selection",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")


# ============================================================
# API URL
# ============================================================

default_api_url = os.getenv(
    "API_URL",
    f"http://localhost:{settings.PORT}",
).rstrip("/")


# ============================================================
# SESSION MANAGER
# ============================================================

st.sidebar.subheader("📂 Session Manager")

try:
    sess_resp = requests.get(
        f"{default_api_url}/sessions",
        timeout=5,
    ).json()

    all_sessions = sess_resp.get(
        "sessions",
        [],
    )
except Exception:
    all_sessions = []

session_map = {
    f"{s['name']} ({s['message_count']} msgs)": s["session_id"]
    for s in all_sessions
}

session_ids = list(session_map.values())
session_labels = list(session_map.keys())

current_idx = 0

if st.session_state.active_session_id in session_ids:
    current_idx = session_ids.index(
        st.session_state.active_session_id
    )

if session_labels:
    selected_label = st.sidebar.selectbox(
        "Active Session:",
        options=session_labels,
        index=current_idx if current_idx < len(session_labels) else 0,
    )

    selected_session_id = session_map.get(
        selected_label,
        "default",
    )

    st.session_state.active_session_id = selected_session_id
else:
    st.sidebar.caption("No saved sessions found.")

col_s1, col_s2 = st.sidebar.columns(2)

with col_s1:
    if st.button(
        "➕ New",
        use_container_width=True,
    ):
        try:
            response = requests.post(
                f"{default_api_url}/sessions",
                timeout=5,
            )

            response.raise_for_status()

            new_s = response.json()["session"]

            st.session_state.active_session_id = new_s[
                "session_id"
            ]

        except Exception:
            st.sidebar.error(
                "Failed to create session: Backend offline."
            )

        st.rerun()

with col_s2:
    if st.button(
        "🗑️ Delete",
        use_container_width=True,
    ):
        try:
            requests.delete(
                f"{default_api_url}/sessions/"
                f"{st.session_state.active_session_id}",
                timeout=5,
            )

            st.session_state.active_session_id = "default"

        except Exception:
            st.sidebar.error(
                "Failed to delete session: Backend offline."
            )

        st.rerun()

st.sidebar.caption(
    f"Session ID: `{st.session_state.active_session_id}`"
)

st.sidebar.markdown("---")


# ============================================================
# SETTINGS
# ============================================================

st.sidebar.subheader("⚙️ Settings & Connection")

theme_mode = st.sidebar.selectbox(
    "🎨 UI Theme Mode",
    options=[
        "🌙 Dark Mode",
        "☀️ Light Mode",
    ],
    index=(
        0
        if st.session_state.get(
            "ui_theme",
            "dark",
        )
        == "dark"
        else 1
    ),
    key="theme_mode_selector",
)

selected_theme = (
    "dark"
    if "Dark" in theme_mode
    else "light"
)

st.session_state.ui_theme = selected_theme

inject_custom_css(selected_theme)

api_base_url = st.sidebar.text_input(
    "FastAPI Server URL",
    value=default_api_url,
).rstrip("/")

top_k_val = st.sidebar.slider(
    "Retrieval Top-K Chunks",
    min_value=1,
    max_value=15,
    value=4,
)

col_ic1, col_ic2 = st.sidebar.columns(2)

with col_ic1:
    user_avatar = st.text_input(
        "User Icon",
        value=settings.USER_AVATAR,
        help="Emoji or image URL",
    )

with col_ic2:
    ai_avatar = st.text_input(
        "AI Icon",
        value=settings.AI_AVATAR,
        help="Emoji or image URL",
    )


# ============================================================
# BACKEND CONNECTION CHECK
# ============================================================

is_connected = False

try:
    r = requests.get(
        f"{api_base_url}/health",
        timeout=2,
    )

    if r.status_code == 200:
        is_connected = True

        st.sidebar.markdown(
            '<div class="pulse-online">'
            '<div class="pulse-dot"></div> '
            "REST API Connected"
            "</div>",
            unsafe_allow_html=True,
        )

    else:
        st.sidebar.warning(
            f"⚠️ API status: {r.status_code}"
        )

except Exception:
    st.sidebar.error(
        "🔴 Backend Offline. "
        "Run `python app.py --backend`"
    )


# ============================================================
# HEADER
# ============================================================

status_badge = (
    '<div class="pulse-online">'
    '<div class="pulse-dot"></div> Online'
    "</div>"
    if is_connected
    else (
        '<span style="color:#f43f5e;'
        'font-weight:600;">Offline</span>'
    )
)

st.html(f"""
    <div class="hero-banner">
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
        ">
            <div>
                <div class="hero-title">
                    ⚡ {settings.PROJECT_NAME}
                </div>

                <div class="hero-subtitle">
                    High-performance Retrieval-Augmented Generation
                    powered by FAISS & Gemini LLMs
                </div>
            </div>

            <div>
                {status_badge}
            </div>
        </div>
    </div>
    """)


# ============================================================
# HELPERS
# ============================================================

def get_active_history():
    """Fetch active session conversation history."""

    try:
        r = requests.get(
            f"{api_base_url}/sessions/"
            f"{st.session_state.active_session_id}",
            timeout=3,
        )

        if r.status_code == 200:
            return r.json().get(
                "history",
                [],
            )

    except Exception:
        pass

    return []


def stream_rag_tokens(
    user_query,
    top_k,
):
    """Stream real-time RAG response tokens."""

    payload = {
        "query": user_query,
        "top_k": top_k,
        "session_id": st.session_state.active_session_id,
    }

    sources_holder = []

    try:
        response = requests.post(
            f"{api_base_url}/query/stream",
            json=payload,
            stream=True,
            timeout=45,
        )

        if response.status_code == 200:

            for line in response.iter_lines():

                if not line:
                    continue

                decoded = line.decode(
                    "utf-8"
                )

                if not decoded.startswith(
                    "data: "
                ):
                    continue

                json_str = decoded[6:]

                try:
                    data = json.loads(
                        json_str
                    )

                    if data["type"] == "sources":

                        sources_holder.extend(
                            data.get(
                                "sources",
                                [],
                            )
                        )

                    elif data["type"] == "token":

                        yield data.get(
                            "content",
                            "",
                        )

                except Exception:
                    pass

        else:

            yield (
                "⚠️ Backend service unavailable "
                f"(HTTP {response.status_code}). "
                "Please check the FastAPI connection."
            )

    except Exception:

        yield (
            "⚠️ Connection Error: Unable to reach "
            f"FastAPI backend at `{api_base_url}`. "
            "Please ensure the backend server is running."
        )

    st.session_state[
        "latest_sources"
    ] = sources_holder


def display_document_sources(
    sources,
    title="📚 Retrieved Context Sources",
):
    """Display ordinary document/RAG source chunks."""

    if not sources:
        return

    with st.expander(
        f"{title} ({len(sources)} chunks)"
    ):

        for idx, src in enumerate(
            sources
        ):

            score = src.get(
                "similarity_score"
            )

            if score is None:

                dist = src.get(
                    "distance",
                    0.0,
                )

                score = 1.0 / (
                    1.0 + dist
                )

            text = src.get(
                "text",
                "",
            )

            meta = src.get(
                "metadata",
                {},
            )

            if isinstance(
                meta,
                dict,
            ):
                source_file = (
                    meta.get(
                        "filename"
                    )
                    or meta.get(
                        "source",
                        "Document",
                    )
                )

                page_num = meta.get(
                    "page_number"
                )

                chunk_id = meta.get(
                    "chunk_id"
                )

            else:
                source_file = "Document"
                page_num = None
                chunk_id = None

            page_badge = (
                f" &bull; <span>"
                f"Page {page_num}"
                f"</span>"
                if page_num is not None
                else ""
            )

            chunk_badge = (
                f" &bull; <span>"
                f"Chunk #{chunk_id}"
                f"</span>"
                if chunk_id is not None
                else ""
            )

            st.html(f"""
                <div class="source-box">
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:6px;
                    ">
                        <strong>
                            Chunk #{idx + 1}
                            &bull;
                            <code>{source_file}</code>
                            {page_badge}
                            {chunk_badge}
                        </strong>

                        <span class="score-meter">
                            Similarity: {score:.3f}
                        </span>
                    </div>
                </div>
                """)

            st.code(
                text,
                language="markdown",
            )


def get_response_detail(response):
    """Extract readable FastAPI error detail."""

    try:
        return response.json().get(
            "detail",
            response.text,
        )
    except Exception:
        return response.text


# ============================================================
# MAIN CONTENT
# ============================================================


# ------------------------------------------------------------
# AI AGENT
# ------------------------------------------------------------

if nav_page == "🤖 NexaMind AI Agent":

    st.subheader(
        "🤖 NexaMind Autonomous AI Agent"
    )

    st.caption(
        "Multi-Tool AI Agent that dynamically routes queries "
        "between Document RAG, YouTube RAG, GitHub RAG, "
        "and live Web Search."
    )

    st.markdown(
        "#### 🛠️ Enable Agent Tools"
    )

    t_c1, t_c2, t_c3, t_c4 = st.columns(
        4
    )

    with t_c1:
        use_doc_rag = st.checkbox(
            "📄 Document RAG",
            value=True,
            help=(
                "Search indexed local "
                "PDFs/documents in FAISS"
            ),
        )

    with t_c2:
        use_yt_rag = st.checkbox(
            "📹 YouTube RAG",
            value=True,
            help=(
                "Search YouTube "
                "transcripts"
            ),
        )

    with t_c3:
        use_github_rag = st.checkbox(
            "💻 GitHub RAG",
            value=True,
            help=(
                "Search source code inside "
                "GitHub repositories"
            ),
        )

    with t_c4:
        use_web_search = st.checkbox(
            "🌐 Web Search",
            value=True,
            help=(
                "Search current public "
                "web information"
            ),
        )

    enabled_tools = []

    if use_doc_rag:
        enabled_tools.append(
            "document_rag"
        )

    if use_yt_rag:
        enabled_tools.append(
            "youtube_rag"
        )

    if use_github_rag:
        enabled_tools.append(
            "github_rag"
        )

    if use_web_search:
        enabled_tools.append(
            "web_search"
        )

    st.markdown(
        "<hr style='border-color:"
        "rgba(255,255,255,0.08)'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**💡 Quick Agent Suggestions:**"
    )

    p1, p2, p3 = st.columns(
        3
    )

    agent_preset = None

    if p1.button(
        "📑 Docs + Web: Compare resume skills with market trends"
    ):
        agent_preset = (
            "What skills are mentioned in the uploaded "
            "documents and what are the current top "
            "in-demand skills on the web?"
        )

    if p2.button(
        "📹 YouTube + Web: Summarize video & search web"
    ):
        agent_preset = (
            "Summarize the key points spoken in video "
            "J5_-l7WIO_w and search the web for related concepts."
        )

    if p3.button(
        "🌐 Web Search: Latest developments in AI agents"
    ):
        agent_preset = (
            "Search the web for the latest developments "
            "in AI agents and LLM tool calling."
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    if (
        "agent_history"
        not in st.session_state
        or st.session_state.get(
            "active_agent_session_id"
        )
        != st.session_state.active_session_id
    ):

        st.session_state[
            "active_agent_session_id"
        ] = st.session_state.active_session_id

        st.session_state[
            "agent_history"
        ] = []

        hist = get_active_history()

        for item in hist:

            if (
                "query" in item
                and "summary" in item
            ):

                st.session_state[
                    "agent_history"
                ].append(
                    {
                        "role": "user",
                        "content": item[
                            "query"
                        ],
                    }
                )

                st.session_state[
                    "agent_history"
                ].append(
                    {
                        "role": "assistant",
                        "content": item[
                            "summary"
                        ],
                        "sources": item.get(
                            "sources",
                            [],
                        ),
                    }
                )

            elif (
                "role" in item
                and "content" in item
            ):

                st.session_state[
                    "agent_history"
                ].append(
                    {
                        "role": item[
                            "role"
                        ],
                        "content": item[
                            "content"
                        ],
                        "sources": item.get(
                            "sources",
                            [],
                        ),
                        "steps": item.get(
                            "steps",
                            [],
                        ),
                        "execution_time_ms": item.get(
                            "execution_time_ms",
                            0,
                        ),
                    }
                )

    with st.container(
        height=420
    ):

        if not st.session_state[
            "agent_history"
        ]:

            st.info(
                "👋 Ask any question. NexaMind will automatically "
                "choose between Document RAG, YouTube RAG, "
                "GitHub RAG, and Web Search."
            )

        else:

            for msg in st.session_state[
                "agent_history"
            ]:

                with st.chat_message(
                    msg["role"],
                    avatar=(
                        user_avatar
                        if msg["role"]
                        == "user"
                        else ai_avatar
                    ),
                ):

                    st.markdown(
                        msg["content"]
                    )

                    if msg.get(
                        "steps"
                    ):

                        with st.expander(
                            "🧩 Agent Tool Calls "
                            f"({len(msg['steps'])} tools executed "
                            f"in {msg.get('execution_time_ms', 0):.0f}ms)"
                        ):

                            tool_icons = {
                                "document_rag": "📄",
                                "youtube_rag": "📹",
                                "github_rag": "💻",
                                "web_search": "🌐",
                            }

                            for idx, step in enumerate(
                                msg["steps"]
                            ):

                                t_icon = tool_icons.get(
                                    step["tool"],
                                    "🛠️",
                                )

                                st.markdown(
                                    f"**Step #{idx + 1} "
                                    f"{t_icon} "
                                    f"`{step['tool']}`** "
                                    f"(Time: "
                                    f"`{step['execution_time_ms']}ms`)"
                                )

                                st.markdown(
                                    f"Input: `{step['input']}`"
                                )

                                st.code(
                                    step[
                                        "output"
                                    ],
                                    language="markdown",
                                )

    agent_query_input = st.chat_input(
        "Ask NexaMind Agent "
        "(Documents, YouTube, GitHub, Web)..."
    )

    final_agent_query = (
        agent_preset
        or agent_query_input
    )

    if final_agent_query:

        st.session_state[
            "agent_history"
        ].append(
            {
                "role": "user",
                "content": final_agent_query,
            }
        )

        with st.spinner(
            "🤖 NexaMind Agent is reasoning "
            "and invoking tools..."
        ):

            try:

                resp = requests.post(
                    f"{api_base_url}/agent/query",
                    json={
                        "query": final_agent_query,
                        "session_id": (
                            st.session_state.active_session_id
                        ),
                        "enabled_tools": enabled_tools,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:

                    data = resp.json()

                    st.session_state[
                        "agent_history"
                    ].append(
                        {
                            "role": "assistant",
                            "content": data[
                                "answer"
                            ],
                            "steps": data.get(
                                "steps",
                                [],
                            ),
                            "sources": data.get(
                                "sources",
                                [],
                            ),
                            "execution_time_ms": data.get(
                                "execution_time_ms",
                                0,
                            ),
                        }
                    )

                else:
                    st.error(
                        "Agent query failed: "
                        f"{get_response_detail(resp)}"
                    )

                    if (
                        st.session_state[
                            "agent_history"
                        ]
                        and st.session_state[
                            "agent_history"
                        ][-1]["role"]
                        == "user"
                    ):
                        st.session_state[
                            "agent_history"
                        ].pop()

            except Exception as ex:

                st.error(
                    "⚠️ Connection Error: "
                    "Unable to execute agent query "
                    f"via FastAPI backend "
                    f"(`{api_base_url}`). "
                    f"Details: {ex!s}"
                )

                if (
                    st.session_state[
                        "agent_history"
                    ]
                    and st.session_state[
                        "agent_history"
                    ][-1]["role"]
                    == "user"
                ):
                    st.session_state[
                        "agent_history"
                    ].pop()

        st.rerun()


# ------------------------------------------------------------
# INTERACTIVE DOCUMENT RAG
# ------------------------------------------------------------

elif nav_page == "💬 Interactive RAG":

    col_h1, col_h2 = st.columns(
        [3, 1]
    )

    with col_h1:

        st.markdown(
            f'<div class="session-badge">'
            f"⚡ Active Session: "
            f"{st.session_state.active_session_id}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_h2:

        if st.button(
            "🗑️ Clear History",
            key="clear_chat_tab_btn",
            use_container_width=True,
        ):

            try:
                requests.delete(
                    f"{api_base_url}/sessions/"
                    f"{st.session_state.active_session_id}",
                    timeout=5,
                )
            except Exception:
                st.error(
                    "Failed to clear session history: "
                    "Backend service unavailable."
                )

            st.rerun()

    st.markdown(
        "**💡 Quick Question Suggestions:**"
    )

    cols_btn = st.columns(
        3
    )

    preset_query = None

    if cols_btn[0].button(
        "🎓 Where did Annanya study?"
    ):
        preset_query = (
            "Where did Annanya study?"
        )

    if cols_btn[1].button(
        "💻 What are key technical skills?"
    ):
        preset_query = (
            "What are the key technical skills "
            "mentioned in the documents?"
        )

    if cols_btn[2].button(
        "📄 Summarize all loaded documents"
    ):
        preset_query = (
            "Summarize the key information "
            "in all available documents."
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    chat_history = get_active_history()

    for chat in chat_history:

        if (
            "query" in chat
            and "summary" in chat
        ):

            with st.chat_message(
                "user",
                avatar=user_avatar,
            ):
                st.write(
                    chat["query"]
                )

            with st.chat_message(
                "assistant",
                avatar=ai_avatar,
            ):
                st.markdown(
                    chat["summary"]
                )

                display_document_sources(
                    chat.get(
                        "sources",
                        [],
                    )
                )

        else:

            role = chat.get(
                "role",
                "user",
            )

            content = (
                chat.get(
                    "content"
                )
                or chat.get(
                    "query"
                )
                or chat.get(
                    "summary"
                )
                or ""
            )

            with st.chat_message(
                role,
                avatar=(
                    user_avatar
                    if role == "user"
                    else ai_avatar
                ),
            ):

                st.markdown(
                    content
                )

                if role == "assistant":
                    display_document_sources(
                        chat.get(
                            "sources",
                            [],
                        )
                    )

    user_input = st.chat_input(
        "Ask any question from your "
        "document knowledge base..."
    )

    final_query = (
        preset_query
        or user_input
    )

    if final_query:

        with st.chat_message(
            "user",
            avatar=user_avatar,
        ):
            st.write(
                final_query
            )

        with st.chat_message(
            "assistant",
            avatar=ai_avatar,
        ):

            st.write_stream(
                stream_rag_tokens(
                    final_query,
                    top_k_val,
                )
            )

            latest_sources = st.session_state.get(
                "latest_sources",
                [],
            )

            display_document_sources(
                latest_sources
            )

        st.rerun()


# ------------------------------------------------------------
# YOUTUBE Q&A
# ------------------------------------------------------------

elif nav_page == "📹 YouTube Q&A":

    st.subheader(
        "📹 YouTube Video Transcript & Q&A"
    )

    st.caption(
        "Extract YouTube transcripts with timestamps, "
        "view video content, and index into NexaMind "
        "vector store for AI Q&A."
    )

    yt_col1, yt_col2 = st.columns(
        [3, 1]
    )

    with yt_col1:

        yt_url = st.text_input(
            "Enter YouTube Video URL or Video ID:",
            placeholder=(
                "https://www.youtube.com/"
                "watch?v=jNQXAC9IVRw"
            ),
            key="yt_url_input",
        )

    with yt_col2:

        auto_index_yt = st.checkbox(
            "Auto-index into Vector Store",
            value=True,
            key="yt_auto_index",
        )

    col_btn1, col_btn2 = st.columns(
        [1, 1]
    )

    with col_btn1:

        fetch_btn = st.button(
            "🚀 Fetch Transcript & Index",
            use_container_width=True,
            key="fetch_yt_btn",
        )

    if (
        fetch_btn
        and yt_url.strip()
    ):

        with st.spinner(
            "Extracting transcript from YouTube..."
        ):

            try:

                resp = requests.post(
                    f"{api_base_url}/youtube/transcript",
                    json={
                        "url": yt_url.strip(),
                        "save_to_dataset": True,
                        "auto_reindex": auto_index_yt,
                    },
                    timeout=45,
                )

                if resp.status_code == 200:

                    st.session_state[
                        "yt_data"
                    ] = resp.json()

                    st.success(
                        "Transcript fetched successfully! "
                        f"({st.session_state['yt_data']['segment_count']} "
                        "segments)"
                    )

                else:

                    st.error(
                        "Failed to fetch transcript via API: "
                        f"{get_response_detail(resp)}"
                    )

            except Exception as ex:

                st.error(
                    "⚠️ Connection Error: "
                    "Unable to fetch YouTube transcript "
                    f"via FastAPI backend (`{api_base_url}`). "
                    f"Details: {ex!s}"
                )

    if st.session_state.get(
        "yt_data"
    ):

        yt_data = st.session_state[
            "yt_data"
        ]

        st.markdown(
            "<hr style='border-color:"
            "rgba(255,255,255,0.08)'>",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(
            3
        )

        with m1:
            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {yt_data['video_id']}
                    </div>
                    <div class="glass-label">
                        Video ID
                    </div>
                </div>
                """)

        with m2:
            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {yt_data['segment_count']}
                    </div>
                    <div class="glass-label">
                        Transcript Segments
                    </div>
                </div>
                """)

        with m3:

            sf_name = yt_data.get(
                "saved_file",
                "Indexed",
            )

            st.html(f"""
                <div class="glass-card">
                    <div
                        class="glass-value"
                        style="
                            color:#34d399;
                            font-size:1.1rem;
                        "
                    >
                        {sf_name}
                    </div>
                    <div class="glass-label">
                        Dataset File Status
                    </div>
                </div>
                """)

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        t_tab1, t_tab2, t_tab3 = st.tabs(
            [
                "⏱️ Timestamped Transcript",
                "📝 Plain Text & Download",
                "⚡ Ask Video Questions",
            ]
        )

        with t_tab1:

            st.video(
                yt_data["url"]
            )

            st.markdown(
                "#### Timestamped Captions:"
            )

            with st.container(
                height=350
            ):

                for seg in yt_data.get(
                    "segments",
                    [],
                ):

                    st.markdown(
                        f"**`[{seg['timestamp']}]`** "
                        f"{seg['text']}"
                    )

        with t_tab2:

            st.markdown(
                "#### Complete Formatted Transcript:"
            )

            st.text_area(
                "Transcript Content:",
                value=yt_data.get(
                    "full_text",
                    "",
                ),
                height=300,
                key="yt_full_text_area",
            )

            st.download_button(
                label="📥 Download Transcript (.txt)",
                data=yt_data.get(
                    "full_text",
                    "",
                ),
                file_name=(
                    f"transcript_"
                    f"{yt_data['video_id']}.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

        with t_tab3:

            st.markdown(
                "#### 💬 Interactive Video Chat & Q&A"
            )

            st.caption(
                "Ask any question specifically about this "
                "video transcript. The AI will answer based "
                "on the video content."
            )

            if (
                "yt_chat_history"
                not in st.session_state
            ):
                st.session_state[
                    "yt_chat_history"
                ] = []

            st.markdown(
                "**💡 Quick Suggestions:**"
            )

            (
                p_col1,
                p_col2,
                p_col3,
                p_col4,
            ) = st.columns(
                [1, 1, 1, 1]
            )

            selected_preset = None

            if p_col1.button(
                "🎯 Summarize Video",
                key="yt_btn_sum",
                use_container_width=True,
            ):
                selected_preset = (
                    "Summarize the main key points "
                    f"of YouTube video "
                    f"{yt_data['video_id']}."
                )

            if p_col2.button(
                "🔑 Key Takeaways",
                key="yt_btn_takeaways",
                use_container_width=True,
            ):
                selected_preset = (
                    "What are the top 3-5 key "
                    "takeaways from video "
                    f"{yt_data['video_id']}?"
                )

            if p_col3.button(
                "🛠️ Methods & Steps",
                key="yt_btn_methods",
                use_container_width=True,
            ):
                selected_preset = (
                    "What processes, tools, or "
                    "steps were discussed in video "
                    f"{yt_data['video_id']}?"
                )

            if p_col4.button(
                "🗑️ Clear Chat",
                key="yt_btn_clear",
                use_container_width=True,
            ):

                st.session_state[
                    "yt_chat_history"
                ] = []

                st.rerun()

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            with st.container(
                height=380
            ):

                if not st.session_state[
                    "yt_chat_history"
                ]:

                    st.info(
                        "👋 Ask any question below or click "
                        "a quick suggestion to start chatting "
                        "about this video!"
                    )

                else:

                    for msg in st.session_state[
                        "yt_chat_history"
                    ]:

                        with st.chat_message(
                            msg["role"]
                        ):

                            st.markdown(
                                msg["content"]
                            )

                            if msg.get(
                                "sources"
                            ):

                                with st.expander(
                                    "📚 Context Chunks Used "
                                    f"({len(msg['sources'])})"
                                ):

                                    for idx, src in enumerate(
                                        msg[
                                            "sources"
                                        ]
                                    ):

                                        score = src.get(
                                            "similarity_score"
                                        )

                                        if score is None:

                                            dist = src.get(
                                                "distance",
                                                0.0,
                                            )

                                            score = 1.0 / (
                                                1.0
                                                + dist
                                            )

                                        meta = src.get(
                                            "metadata",
                                            {},
                                        )

                                        ts = (
                                            meta.get(
                                                "timestamp"
                                            )
                                            or src.get(
                                                "timestamp"
                                            )
                                            or "00:00"
                                        )

                                        deep_link = (
                                            meta.get(
                                                "url"
                                            )
                                            or src.get(
                                                "url"
                                            )
                                            or ""
                                        )

                                        st.markdown(
                                            f"**Chunk #{idx + 1}** "
                                            f"| Similarity: "
                                            f"`{score:.3f}` "
                                            f"| Timestamp: `{ts}`"
                                        )

                                        if deep_link:
                                            st.markdown(
                                                f"[▶️ Watch segment]"
                                                f"({deep_link})"
                                            )

                                        st.code(
                                            src.get(
                                                "text",
                                                "",
                                            ),
                                            language="markdown",
                                        )

            yt_user_query = st.chat_input(
                "Ask a question about this video...",
                key="yt_tab_chat_input",
            )

            final_yt_query = (
                selected_preset
                or yt_user_query
            )

            if final_yt_query:

                st.session_state[
                    "yt_chat_history"
                ].append(
                    {
                        "role": "user",
                        "content": final_yt_query,
                    }
                )

                with st.chat_message(
                    "assistant"
                ):

                    full_response = st.write_stream(
                        stream_rag_tokens(
                            final_yt_query,
                            top_k=5,
                        )
                    )

                    latest_srcs = st.session_state.get(
                        "latest_sources",
                        [],
                    )

                    st.session_state[
                        "yt_chat_history"
                    ].append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "sources": latest_srcs,
                        }
                    )

                st.rerun()


# ------------------------------------------------------------
# GITHUB Q&A
# ------------------------------------------------------------

elif nav_page == "💻 GitHub Q&A":

    st.subheader(
        "💻 GitHub Repository Q&A"
    )

    st.caption(
        "Connect a public GitHub repository, build a "
        "repository-specific FAISS index, and ask "
        "questions about its source code."
    )

    github_repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder=(
            "https://github.com/"
            "annanyasinha/localconnect"
        ),
        key="github_repo_url",
    )

    index_button = st.button(
        "🚀 Load / Index Repository",
        use_container_width=True,
        key="github_index_button",
    )

    if index_button:

        if not github_repo_url.strip():

            st.warning(
                "Please enter a GitHub repository URL."
            )

        else:

            with st.spinner(
                "Checking repository and preparing "
                "FAISS index..."
            ):

                try:

                    response = requests.post(
                        f"{api_base_url}/github/index",
                        json={
                            "repo_url": (
                                github_repo_url.strip()
                            )
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state[
                            "github_repo_data"
                        ] = data

                        st.session_state[
                            "active_github_repo"
                        ] = github_repo_url.strip()

                        # Clear old answer whenever a repo is indexed/loaded
                        st.session_state[
                            "github_last_answer"
                        ] = None

                        if data.get(
                            "cached"
                        ):

                            st.success(
                                "✅ Repository loaded from "
                                "existing FAISS cache."
                            )

                        else:

                            st.success(
                                "✅ Repository indexed "
                                "successfully."
                            )

                    else:

                        st.error(
                            "GitHub indexing failed: "
                            f"{get_response_detail(response)}"
                        )

                except Exception as ex:

                    st.error(
                        "⚠️ Unable to connect to FastAPI "
                        f"backend: {ex!s}"
                    )

    github_data = st.session_state.get(
        "github_repo_data"
    )

    if github_data:

        st.markdown(
            "---"
        )

        c1, c2, c3 = st.columns(
            3
        )

        with c1:

            st.metric(
                "Repository",
                github_data.get(
                    "repository",
                    "Unknown",
                ),
            )

        with c2:

            st.metric(
                "Vectors",
                github_data.get(
                    "total_vectors",
                    0,
                ),
            )

        with c3:

            cache_status = (
                "Cached"
                if github_data.get(
                    "cached"
                )
                else "New Index"
            )

            st.metric(
                "Index Status",
                cache_status,
            )

        commit_sha = github_data.get(
            "commit_sha",
            "",
        )

        if commit_sha:

            st.caption(
                "Indexed commit: "
                f"`{commit_sha[:12]}`"
            )

        st.markdown(
            "### 💬 Ask About This Repository"
        )

        github_question = st.text_input(
            "Question",
            placeholder=(
                "How is JWT authentication implemented?"
            ),
            key="github_question",
        )

        ask_github_button = st.button(
            "🔎 Ask Repository",
            use_container_width=True,
            key="github_ask_button",
        )

        if ask_github_button:

            if not github_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                repo_url = st.session_state.get(
                    "active_github_repo"
                )

                if not repo_url:

                    st.warning(
                        "Please load a repository first."
                    )

                else:

                    full_query = (
                        f"{repo_url}\n\n"
                        f"{github_question.strip()}"
                    )

                    with st.spinner(
                        "Searching repository and "
                        "generating answer..."
                    ):

                        try:

                            response = requests.post(
                                f"{api_base_url}/agent/query",
                                json={
                                    "query": full_query,
                                    "session_id": (
                                        st.session_state.active_session_id
                                    ),
                                    "enabled_tools": [
                                        "github_rag"
                                    ],
                                },
                                timeout=120,
                            )

                            if response.status_code == 200:

                                result = response.json()

                                st.session_state[
                                    "github_last_answer"
                                ] = result

                            else:

                                st.error(
                                    "GitHub Q&A failed: "
                                    f"{get_response_detail(response)}"
                                )

                        except Exception as ex:

                            st.error(
                                "⚠️ Unable to execute "
                                "GitHub question: "
                                f"{ex!s}"
                            )

        github_answer = st.session_state.get(
            "github_last_answer"
        )

        if github_answer:

            st.markdown(
                "### 🤖 Answer"
            )

            st.markdown(
                github_answer.get(
                    "answer",
                    "No answer generated.",
                )
            )

            steps = github_answer.get(
                "steps",
                [],
            )

            if steps:

                with st.expander(
                    "🧩 Agent Execution"
                ):

                    for idx, step in enumerate(
                        steps
                    ):

                        st.markdown(
                            f"**Step #{idx + 1}: "
                            f"`{step.get('tool', 'unknown')}`** "
                            f"({step.get('execution_time_ms', 0)} ms)"
                        )

            sources = github_answer.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    "💻 GitHub Sources "
                    f"({len(sources)} chunks)"
                ):

                    for index, source in enumerate(
                        sources
                    ):

                        file_path = (
                            source.get(
                                "file_path"
                            )
                            or source.get(
                                "source"
                            )
                            or (
                                source.get(
                                    "metadata",
                                    {},
                                ).get(
                                    "file_path"
                                )
                                if isinstance(
                                    source.get(
                                        "metadata",
                                        {},
                                    ),
                                    dict,
                                )
                                else None
                            )
                            or "Unknown file"
                        )

                        score = source.get(
                            "similarity_score",
                            0.0,
                        )

                        github_url = (
                            source.get(
                                "url"
                            )
                            or source.get(
                                "github_url"
                            )
                            or (
                                source.get(
                                    "metadata",
                                    {},
                                ).get(
                                    "github_url"
                                )
                                if isinstance(
                                    source.get(
                                        "metadata",
                                        {},
                                    ),
                                    dict,
                                )
                                else ""
                            )
                        )

                        language = (
                            source.get(
                                "language"
                            )
                            or (
                                source.get(
                                    "metadata",
                                    {},
                                ).get(
                                    "language"
                                )
                                if isinstance(
                                    source.get(
                                        "metadata",
                                        {},
                                    ),
                                    dict,
                                )
                                else "text"
                            )
                            or "text"
                        )

                        text = source.get(
                            "text",
                            "",
                        )

                        st.markdown(
                            f"**{index + 1}. "
                            f"`{file_path}`**"
                        )

                        st.caption(
                            f"Similarity: {score:.3f}"
                        )

                        if github_url:

                            st.markdown(
                                f"[View file on GitHub]"
                                f"({github_url})"
                            )

                        if text:

                            st.code(
                                text,
                                language=language,
                            )


# ------------------------------------------------------------
# VECTOR EXPLORER
# ------------------------------------------------------------

elif nav_page == "🔍 Vector Explorer":

    st.subheader(
        "🔍 Vector Similarity Explorer"
    )

    st.caption(
        "Inspect raw FAISS vector similarity scores "
        "and extracted text chunks without LLM summarization."
    )

    col_v1, col_v2 = st.columns(
        [3, 1]
    )

    with col_v1:

        vec_query = st.text_input(
            "Enter search query for raw similarity match:",
            value="Shubham education details",
        )

    with col_v2:

        vec_top_k = st.slider(
            "Top K matches:",
            1,
            10,
            5,
            key="vec_top_k",
        )

    if st.button(
        "🔍 Search FAISS Index",
        use_container_width=True,
    ):

        if vec_query.strip():

            with st.spinner(
                "Executing FAISS vector search..."
            ):

                try:

                    resp = requests.post(
                        f"{api_base_url}/search",
                        json={
                            "query": vec_query,
                            "top_k": vec_top_k,
                        },
                        timeout=30,
                    )

                    resp.raise_for_status()

                    results = resp.json().get(
                        "results",
                        [],
                    )

                    st.success(
                        f"Retrieved {len(results)} "
                        "vector matches!"
                    )

                    for idx, res in enumerate(
                        results
                    ):

                        score = res.get(
                            "similarity_score"
                        )

                        if score is None:

                            dist = res.get(
                                "distance",
                                0.0,
                            )

                            score = 1.0 / (
                                1.0 + dist
                            )

                        meta = res.get(
                            "metadata",
                            {},
                        )

                        text = (
                            meta.get(
                                "text",
                                "No text found",
                            )
                            if meta
                            else "N/A"
                        )

                        page_num = (
                            meta.get(
                                "page_number"
                            )
                            if isinstance(
                                meta,
                                dict,
                            )
                            else None
                        )

                        fn = (
                            meta.get(
                                "filename"
                            )
                            or meta.get(
                                "source",
                                "",
                            )
                        )

                        col1, col2 = st.columns(
                            [1, 4]
                        )

                        with col1:

                            st.html(f"""
                                <div class="glass-card">
                                    <div class="glass-value">
                                        #{idx + 1}
                                    </div>
                                    <div class="glass-label">
                                        Sim: {score:.4f}
                                    </div>
                                    <div style="margin-top:6px;">
                                        <span class="score-meter">
                                            Score: {score:.3f}
                                        </span>
                                    </div>
                                </div>
                                """)

                        with col2:

                            page_text = (
                                f" &bull; Page: "
                                f"`{page_num}`"
                                if page_num
                                else ""
                            )

                            st.markdown(
                                f"**Chunk Index:** "
                                f"`{res.get('index', 'N/A')}` "
                                f"&bull; Document: "
                                f"<code>{fn}</code>"
                                f"{page_text}",
                                unsafe_allow_html=True,
                            )

                            st.text_area(
                                "Extracted Context Text:",
                                value=text,
                                height=120,
                                key=f"raw_text_{idx}",
                            )

                        st.markdown(
                            "<hr style='border-color:"
                            "rgba(255,255,255,0.08)'>",
                            unsafe_allow_html=True,
                        )

                except Exception as e:

                    st.error(
                        f"Vector search failed: {e!s}"
                    )


# ------------------------------------------------------------
# DOCUMENT HUB
# ------------------------------------------------------------

elif nav_page == "📁 Document Hub":

    st.subheader(
        "📁 Document Knowledge Base Manager"
    )

    st.caption(
        "Upload documents to your knowledge dataset "
        "and rebuild vector indexes dynamically."
    )

    col_up, col_list = st.columns(
        [1, 1]
    )

    with col_up:

        st.markdown(
            "#### 📤 Upload Documents"
        )

        uploaded_files = st.file_uploader(
            "Supported file types: "
            "PDF, TXT, CSV, DOCX, XLSX, JSON",
            accept_multiple_files=True,
            type=[
                "pdf",
                "txt",
                "csv",
                "docx",
                "xlsx",
                "json",
            ],
        )

        auto_index_check = st.checkbox(
            "Automatically rebuild vector index after upload",
            value=True,
        )

        if st.button(
            "🚀 Process & Index Files",
            use_container_width=True,
        ):

            if uploaded_files:

                with st.spinner(
                    "Uploading and indexing documents "
                    "via FastAPI..."
                ):

                    files_payload = [
                        (
                            "files",
                            (
                                f.name,
                                f.getvalue(),
                                f.type,
                            ),
                        )
                        for f in uploaded_files
                    ]

                    data_payload = {
                        "auto_reindex": str(
                            auto_index_check
                        ).lower()
                    }

                    try:

                        res = requests.post(
                            f"{api_base_url}/upload",
                            files=files_payload,
                            data=data_payload,
                            timeout=120,
                        )

                        if res.status_code == 200:

                            st.success(
                                res.json().get(
                                    "message",
                                    "Upload successful!",
                                )
                            )

                        else:

                            st.error(
                                "Upload failed via API: "
                                f"{get_response_detail(res)}"
                            )

                    except Exception as ex:

                        st.error(
                            f"Upload failed: {ex!s}"
                        )

            else:

                st.warning(
                    "Please select at least one "
                    "document to upload."
                )

    with col_list:

        st.markdown(
            "#### 📂 Active Document Dataset"
        )

        try:

            docs_res = requests.get(
                f"{api_base_url}/documents",
                timeout=2,
            ).json()

            doc_files = docs_res.get(
                "documents",
                [],
            )

            if doc_files:

                for idx, doc in enumerate(
                    doc_files
                ):

                    fname = doc[
                        "filename"
                    ]

                    fsize = (
                        doc["size_bytes"]
                        / 1024.0
                    )

                    ext = doc.get(
                        "extension",
                        "",
                    ).upper()

                    c_doc1, c_doc2 = st.columns(
                        [4, 1]
                    )

                    with c_doc1:

                        st.html(f"""
                            <div style="
                                background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.08);
                                padding:8px 14px;
                                border-radius:10px;
                                display:flex;
                                justify-content:space-between;
                                align-items:center;
                            ">
                                <span>
                                    📄
                                    <strong>
                                        <code>{fname}</code>
                                    </strong>
                                </span>

                                <span style="
                                    font-size:0.8rem;
                                    color:#9ca3af;
                                ">
                                    {fsize:.1f} KB
                                    &bull;
                                    <b style="color:#34d399">
                                        {ext}
                                    </b>
                                </span>
                            </div>
                            """)

                    with c_doc2:

                        if st.button(
                            "🗑️",
                            key=f"del_doc_{idx}",
                            help=(
                                f"Remove '{fname}' "
                                "from dataset"
                            ),
                        ):

                            try:

                                r_del = requests.delete(
                                    f"{api_base_url}/documents/"
                                    f"{fname}",
                                    timeout=30,
                                )

                                if r_del.status_code == 200:

                                    st.toast(
                                        f"Deleted '{fname}'",
                                        icon="🗑️",
                                    )

                                    st.rerun()

                                else:

                                    st.error(
                                        "Failed to delete: "
                                        f"{get_response_detail(r_del)}"
                                    )

                            except Exception:

                                st.error(
                                    f"Failed to delete '{fname}': "
                                    "Backend service is unavailable."
                                )

            else:

                st.info(
                    "No documents uploaded yet."
                )

        except Exception:

            st.info(
                "Unable to fetch document list "
                "from REST API."
            )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        col_b1, col_b2 = st.columns(
            2
        )

        with col_b1:

            if st.button(
                "🔄 Rebuild Index",
                use_container_width=True,
            ):

                with st.spinner(
                    "Rebuilding FAISS vector index..."
                ):

                    try:

                        r = requests.post(
                            f"{api_base_url}/reindex",
                            timeout=120,
                        )

                        st.success(
                            r.json().get(
                                "message",
                                "Reindexed successfully!",
                            )
                        )

                    except Exception as e:

                        st.error(
                            f"Reindex failed: {e!s}"
                        )

        with col_b2:

            if st.button(
                "🗑️ Clear All Data",
                use_container_width=True,
                help=(
                    "Remove all documents and "
                    "wipe vector store"
                ),
            ):

                with st.spinner(
                    "Wiping dataset & clearing "
                    "vector index..."
                ):

                    try:

                        r_clr = requests.delete(
                            f"{api_base_url}/documents",
                            timeout=30,
                        )

                        if r_clr.status_code == 200:

                            st.success(
                                "All ingested data "
                                "cleared successfully!"
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Failed to clear documents: "
                                f"{get_response_detail(r_clr)}"
                            )

                    except Exception as e:

                        st.error(
                            f"Clear failed: {e!s}"
                        )


# ------------------------------------------------------------
# SYSTEM DASHBOARD
# ------------------------------------------------------------

elif nav_page == "⚙️ System Dashboard":

    st.subheader(
        "⚙️ System Monitor & REST Endpoints"
    )

    st.caption(
        "Live operational state and REST API documentation."
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    try:

        h_res = requests.get(
            f"{api_base_url}/health",
            timeout=3,
        ).json()

        with col1:

            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {h_res.get('status', 'unknown').upper()}
                    </div>
                    <div class="glass-label">
                        API Health
                    </div>
                </div>
                """)

        with col2:

            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {h_res.get('total_vectors', 0)}
                    </div>
                    <div class="glass-label">
                        FAISS Vectors
                    </div>
                </div>
                """)

        with col3:

            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {h_res.get('active_sessions', 0)}
                    </div>
                    <div class="glass-label">
                        Active Sessions
                    </div>
                </div>
                """)

        with col4:

            st.html(f"""
                <div class="glass-card">
                    <div class="glass-value">
                        {h_res.get('data_files_count', 0)}
                    </div>
                    <div class="glass-label">
                        Data Documents
                    </div>
                </div>
                """)

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "#### 🌐 Registered REST API Endpoints"
        )

        st.markdown(
            dedent("""
            | Method | Endpoint | Description |
            | :---: | :--- | :--- |
            | <span class="http-get">GET</span> | `/health` | Live system health metrics & vector count |
            | <span class="http-post">POST</span> | `/agent/query` | NexaMind Autonomous AI Agent query with multi-tool routing & traces |
            | <span class="http-post">POST</span> | `/query` | Session-aware RAG search & Gemini AI generation |
            | <span class="http-post">POST</span> | `/query/stream` | Real-time SSE token streaming RAG endpoint |
            | <span class="http-post">POST</span> | `/search` | Raw FAISS vector similarity search |
            | <span class="http-get">GET</span> | `/sessions` | List active chat sessions |
            | <span class="http-post">POST</span> | `/sessions` | Create new chat session |
            | <span class="http-delete">DELETE</span> | `/sessions/{id}` | Clear turn history or delete session |
            | <span class="http-get">GET</span> | `/documents` | List uploaded dataset files |
            | <span class="http-post">POST</span> | `/upload` | Upload & auto-reindex documents |
            | <span class="http-post">POST</span> | `/reindex` | Force rebuild FAISS index |
            | <span class="http-post">POST</span> | `/youtube/transcript` | Extract YouTube transcript & optionally auto-index into FAISS |
            | <span class="http-post">POST</span> | `/github/index` | Index or load a GitHub repository using commit-SHA cache |
            | <span class="http-post">POST</span> | `/github/query` | Direct semantic search over a GitHub repository |
            | <span class="http-delete">DELETE</span> | `/documents/{filename}` | Delete single file & update vector index |
            | <span class="http-delete">DELETE</span> | `/documents` | Clear all dataset files & wipe vector store |
            """),
            unsafe_allow_html=True,
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "📖 **Interactive Swagger UI OpenAPI Documentation:** "
            f"[{api_base_url}/docs]({api_base_url}/docs)"
        )

    except Exception as ex:

        st.error(
            "Unable to reach REST backend at "
            f"{api_base_url}: {ex!s}"
        )
