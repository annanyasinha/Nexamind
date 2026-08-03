import streamlit as st

COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    /* Sidebar Radio Navigation */
    .stSidebar [data-testid="stRadio"] > label { display: none; }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] { gap: 8px; }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label {
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 600;
        transition: all 0.25s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* HTTP Verb Badges */
    .http-get { background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }
    .http-post { background: rgba(56, 189, 248, 0.2); color: #0284c7; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }
    .http-delete { background: rgba(244, 63, 94, 0.2); color: #e11d48; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

DARK_MODE_CSS = COMMON_CSS + """
<style>
    [data-testid="stAppViewContainer"] { background-color: #0b0f19; color: #f3f4f6; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid rgba(255,255,255,0.08); }

    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.95) 50%, rgba(17, 24, 39, 0.95) 100%);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    .hero-title {
        font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #34d399 0%, #38bdf8 50%, #a78bfa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.4rem;
    }
    .hero-subtitle { font-size: 1.05rem; color: #9ca3af; font-weight: 400; }

    .pulse-online {
        display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.15);
        color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    }
    .pulse-dot { width: 8px; height: 8px; background-color: #34d399; border-radius: 50%; box-shadow: 0 0 8px #34d399; }

    .glass-card {
        background: rgba(22, 27, 34, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px;
        padding: 1.3rem 1.1rem; text-align: center; backdrop-filter: blur(12px); transition: all 0.25s ease;
    }
    .glass-card:hover { transform: translateY(-3px); border-color: rgba(52, 211, 153, 0.3); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
    .glass-value { font-size: 1.9rem; font-weight: 800; background: linear-gradient(90deg, #34d399, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .glass-label { font-size: 0.78rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-top: 4px; }

    .session-badge {
        display: inline-flex; align-items: center; gap: 6px; background: rgba(56, 189, 248, 0.15);
        color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 12px; border-radius: 10px; font-size: 0.82rem; font-weight: 600; margin-bottom: 12px;
    }
    .source-box { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid #34d399; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
    .score-meter { display: inline-block; background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700; font-size: 0.78rem; padding: 2px 8px; border-radius: 6px; }

    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(22, 27, 34, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); color: #9ca3af;
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: rgba(52, 211, 153, 0.4); color: #ffffff; background: rgba(30, 41, 59, 0.8); transform: translateX(4px);
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.2) 0%, rgba(56, 189, 248, 0.2) 100%) !important;
        border: 1px solid rgba(52, 211, 153, 0.5) !important; color: #34d399 !important; font-weight: 700; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p { font-size: 0.95rem; font-weight: 600; }
    .stButton>button { border: 1px solid rgba(255, 255, 255, 0.12); color: #f3f4f6; }
    .stButton>button:hover { border-color: #34d399; box-shadow: 0 4px 14px rgba(52, 211, 153, 0.25); }
</style>
"""

LIGHT_MODE_CSS = COMMON_CSS + """
<style>
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }

    .hero-banner {
        background: linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 50%, #f0fdf4 100%);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.05);
    }
    .hero-title {
        font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 50%, #7c3aed 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.4rem;
    }
    .hero-subtitle { font-size: 1.05rem; color: #475569; font-weight: 500; }

    .pulse-online {
        display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.15);
        color: #059669; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    }
    .pulse-dot { width: 8px; height: 8px; background-color: #059669; border-radius: 50%; box-shadow: 0 0 8px #059669; }

    .glass-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1.3rem 1.1rem; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03); transition: all 0.25s ease;
    }
    .glass-card:hover { transform: translateY(-3px); border-color: #0284c7; box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.15); }
    .glass-value { font-size: 1.9rem; font-weight: 800; background: linear-gradient(90deg, #0284c7, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .glass-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-top: 4px; }

    .session-badge {
        display: inline-flex; align-items: center; gap: 6px; background: rgba(2, 132, 199, 0.1);
        color: #0284c7; border: 1px solid rgba(2, 132, 199, 0.3); padding: 4px 12px; border-radius: 10px; font-size: 0.82rem; font-weight: 600; margin-bottom: 12px;
    }
    .source-box { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
    .score-meter { display: inline-block; background: rgba(2, 132, 199, 0.15); color: #0284c7; font-weight: 700; font-size: 0.78rem; padding: 2px 8px; border-radius: 6px; }

    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label {
        background: #ffffff; border: 1px solid #e2e8f0; color: #475569;
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #0284c7; color: #0f172a; background: #f0f9ff; transform: translateX(4px);
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%) !important;
        border: 1px solid #0284c7 !important; color: #0284c7 !important; font-weight: 700; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
    }
    .stSidebar [data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p { font-size: 0.95rem; font-weight: 600; }
    .stButton>button { border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; }
    .stButton>button:hover { border-color: #0284c7; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.2); }
</style>
"""

CUSTOM_CSS = DARK_MODE_CSS

def inject_custom_css(theme: str = "dark"):
    """Injects custom dark-mode or light-mode CSS aesthetics and navigation styles into Streamlit."""
    css_content = DARK_MODE_CSS if str(theme).lower() == "dark" else LIGHT_MODE_CSS
    st.markdown(css_content, unsafe_allow_html=True)
