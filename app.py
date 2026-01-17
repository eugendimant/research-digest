"""
Eugen's Research Digest - Configuration Dashboard
Streamlit app for managing digest settings.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Eugen's Research Digest",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #93c5fd;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    .info-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-card h3 {
        color: #1e40af;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIG FILE HANDLING
# =============================================================================

CONFIG_FILE = "config.json"
SEEN_PAPERS_FILE = "seen_papers.json"

DEFAULT_CONFIG = {
    "email": {
        "recipient": "edimant@sas.upenn.edu",
        "sender_name": "Eugen's Research Digest"
    },
    "schedule": {
        "frequency": "weekly",
        "day_of_week": 6,
        "hour_utc": 23
    },
    "topics": {
        "primary": [
            "behavioral economics",
            "experimental economics",
            "dishonesty",
            "cheating behavior",
            "unethical behavior",
            "corruption",
            "social norms",
            "honesty",
            "deception",
            "moral behavior"
        ],
        "secondary": [
            "nudge",
            "decision making",
            "prosocial behavior",
            "cooperation",
            "trust game",
            "public goods game",
            "field experiment",
            "lab experiment",
            "randomized controlled trial",
            "behavioral intervention",
            "norm enforcement",
            "peer effects"
        ],
        "exclude": [
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "blockchain",
            "stock prediction",
            "algorithmic trading",
            "sports betting"
        ]
    },
    "sources": {
        "crossref": True,
        "openalex": True,
        "semantic_scholar": True,
        "pubmed": True
    },
    "filters": {
        "days_lookback": 14,
        "min_relevance": 0.12,
        "max_papers": 20,
        "skip_seen_papers": True
    }
}

def load_config():
    """Load config from file or return defaults."""
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Merge with defaults to ensure all keys exist
            for key in DEFAULT_CONFIG:
                if key not in config:
                    config[key] = DEFAULT_CONFIG[key]
            return config
    except FileNotFoundError:
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save config to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def load_seen_papers():
    """Load seen papers tracking data."""
    try:
        with open(SEEN_PAPERS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"paper_ids": [], "last_updated": None}

def clear_seen_papers():
    """Clear seen papers history."""
    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump({"paper_ids": [], "last_updated": datetime.now().isoformat()}, f)

def load_latest_papers():
    """Load the most recent papers from last digest."""
    try:
        with open("latest_papers.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Eugen's Research Digest</h1>
        <p>Automated weekly research paper alerts with verified DOIs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load config
    config = load_config()
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/arxiv.svg", width=50)
        st.header("⚡ Quick Actions")
        
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            save_config(st.session_state.get('config', config))
            st.success("✅ Settings saved!")
            st.balloons()
        
        st.divider()
        
        # Stats
        seen_data = load_seen_papers()
        papers_tracked = len(seen_data.get("paper_ids", []))
        last_updated = seen_data.get("last_updated")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers Tracked", papers_tracked)
        with col2:
            st.metric("Sources", sum(config.get("sources", {}).values()))
        
        if last_updated:
            st.caption(f"Last run: {last_updated[:10]}")
        
        st.divider()
        
        if st.button("🗑️ Clear Paper History", use_container_width=True):
            clear_seen_papers()
            st.success("History cleared!")
            st.rerun()
        
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            save_config(DEFAULT_CONFIG)
            st.success("Reset to defaults!")
            st.rerun()
        
        st.divider()
        
        st.markdown("""
        <div class="info-box">
            <strong>✓ DOI Verification</strong><br>
            <small>All papers are verified against Crossref registry. No fake citations.</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📧 Email & Schedule",
        "🔍 Research Topics", 
        "📚 Sources",
        "⚙️ Filters",
        "📊 Recent Papers"
    ])
    
    # =====================
    # TAB 1: Email & Schedule
    # =====================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📧 Email Settings")
            
            config["email"]["recipient"] = st.text_input(
                "Recipient Email",
                value=config.get("email", {}).get("recipient", "edimant@sas.upenn.edu"),
                help="Where the digest will be sent"
            )
            
            config["email"]["sender_name"] = st.text_input(
                "Sender Display Name",
                value=config.get("email", {}).get("sender_name", "Eugen's Research Digest")
            )
            
            st.markdown("""
            <div class="info-box">
                <strong>📌 Note:</strong> SMTP credentials are stored securely in GitHub Secrets, not here.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📅 Schedule")
            
            frequency = st.selectbox(
                "Frequency",
                options=["daily", "weekly", "biweekly", "monthly"],
                index=["daily", "weekly", "biweekly", "monthly"].index(
                    config.get("schedule", {}).get("frequency", "weekly")
                )
            )
            config["schedule"]["frequency"] = frequency
            
            if frequency in ["weekly", "biweekly"]:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_idx = st.selectbox(
                    "Day of Week",
                    options=range(7),
                    format_func=lambda x: days[x],
                    index=config.get("schedule", {}).get("day_of_week", 6)
                )
                config["schedule"]["day_of_week"] = day_idx
            
            hour = st.slider(
                "Time (UTC)",
                min_value=0,
                max_value=23,
                value=config.get("schedule", {}).get("hour_utc", 23),
                help="When to send the digest"
            )
            config["schedule"]["hour_utc"] = hour
            
            # Time conversion
            est_hour = (hour - 5) % 24
            pst_hour = (hour - 8) % 24
            st.info(f"🕐 {hour}:00 UTC = {est_hour}:00 EST = {pst_hour}:00 PST")
            
            # Cron expression helper
            if frequency == "weekly":
                cron = f"0 {hour} * * {day_idx}"
            elif frequency == "daily":
                cron = f"0 {hour} * * *"
            elif frequency == "biweekly":
                cron = f"0 {hour} * * {day_idx}"  # Needs manual adjustment
            else:
                cron = f"0 {hour} 1 * *"
            
            st.code(f"Cron: {cron}", language=None)
    
    # =====================
    # TAB 2: Research Topics
    # =====================
    with tab2:
        st.subheader("🎯 Primary Keywords")
        st.caption("Highest priority - papers with these terms rank highest (65% weight)")
        
        primary = config.get("topics", {}).get("primary", [])
        primary_text = st.text_area(
            "Primary Keywords (one per line)",
            value="\n".join(primary),
            height=250,
            key="primary_keywords"
        )
        config["topics"]["primary"] = [k.strip() for k in primary_text.split("\n") if k.strip()]
        
        st.divider()
        
        st.subheader("📌 Secondary Keywords")
        st.caption("Medium priority - additional relevant terms (35% weight)")
        
        secondary = config.get("topics", {}).get("secondary", [])
        secondary_text = st.text_area(
            "Secondary Keywords (one per line)",
            value="\n".join(secondary),
            height=200,
            key="secondary_keywords"
        )
        config["topics"]["secondary"] = [k.strip() for k in secondary_text.split("\n") if k.strip()]
        
        st.divider()
        
        st.subheader("🚫 Exclude Terms")
        st.caption("Papers containing these terms are filtered out")
        
        exclude = config.get("topics", {}).get("exclude", [])
        exclude_text = st.text_area(
            "Exclude Terms (one per line)",
            value="\n".join(exclude),
            height=120,
            key="exclude_terms"
        )
        config["topics"]["exclude"] = [k.strip() for k in exclude_text.split("\n") if k.strip()]
        
        # Keyword stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Primary", len(config["topics"]["primary"]))
        with col2:
            st.metric("Secondary", len(config["topics"]["secondary"]))
        with col3:
            st.metric("Excluded", len(config["topics"]["exclude"]))
    
    # =====================
    # TAB 3: Sources
    # =====================
    with tab3:
        st.subheader("📚 Academic Sources")
        st.caption("Enable or disable paper sources. All papers are verified against Crossref.")
        
        sources_info = {
            "crossref": {
                "name": "Crossref",
                "desc": "Authoritative DOI registry - primary source for verification",
                "icon": "🔗",
                "required": True
            },
            "openalex": {
                "name": "OpenAlex",
                "desc": "Open catalog of 250M+ scholarly works",
                "icon": "🌐",
                "required": False
            },
            "semantic_scholar": {
                "name": "Semantic Scholar",
                "desc": "AI-powered academic search from Allen Institute",
                "icon": "🤖",
                "required": False
            },
            "pubmed": {
                "name": "PubMed",
                "desc": "Biomedical and psychology literature (NIH)",
                "icon": "🏥",
                "required": False
            }
        }
        
        col1, col2 = st.columns(2)
        
        for i, (key, info) in enumerate(sources_info.items()):
            col = col1 if i % 2 == 0 else col2
            
            with col:
                with st.container():
                    current = config.get("sources", {}).get(key, True)
                    
                    if info["required"]:
                        st.checkbox(
                            f"{info['icon']} **{info['name']}** (Required)",
                            value=True,
                            disabled=True,
                            key=f"source_{key}"
                        )
                        config["sources"][key] = True
                    else:
                        enabled = st.checkbox(
                            f"{info['icon']} **{info['name']}**",
                            value=current,
                            key=f"source_{key}"
                        )
                        config["sources"][key] = enabled
                    
                    st.caption(info["desc"])
                    st.write("")
        
        st.markdown("""
        <div class="info-box">
            <strong>🔒 Verification Process:</strong><br>
            Every paper from any source is cross-checked against Crossref's DOI registry. 
            If the DOI doesn't exist in Crossref, the paper is rejected. This guarantees 
            all links in your digest point to real, accessible papers.
        </div>
        """, unsafe_allow_html=True)
    
    # =====================
    # TAB 4: Filters
    # =====================
    with tab4:
        st.subheader("🔧 Filter Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            days = st.slider(
                "Days Lookback",
                min_value=1,
                max_value=30,
                value=config.get("filters", {}).get("days_lookback", 14),
                help="Search for papers from the last N days"
            )
            config["filters"]["days_lookback"] = days
            
            max_papers = st.slider(
                "Max Papers per Digest",
                min_value=5,
                max_value=50,
                value=config.get("filters", {}).get("max_papers", 20),
                help="Maximum number of papers to include"
            )
            config["filters"]["max_papers"] = max_papers
        
        with col2:
            min_rel = st.slider(
                "Minimum Relevance Score",
                min_value=0.0,
                max_value=0.5,
                value=float(config.get("filters", {}).get("min_relevance", 0.12)),
                step=0.02,
                help="Papers below this score are filtered out"
            )
            config["filters"]["min_relevance"] = min_rel
            
            skip_seen = st.checkbox(
                "Skip Previously Sent Papers",
                value=config.get("filters", {}).get("skip_seen_papers", True),
                help="Only include papers you haven't received"
            )
            config["filters"]["skip_seen_papers"] = skip_seen
        
        st.divider()
        
        st.subheader("📊 Relevance Scoring Explained")
        
        st.markdown("""
        Each paper gets a score from **0.0 to 1.0** based on keyword matches:
        
        | Component | Weight | Description |
        |-----------|--------|-------------|
        | Primary keywords | 65% | Core research terms |
        | Secondary keywords | 35% | Related concepts |
        | Multi-match bonus | +30% | 2+ primary matches |
        
        **Example:** A paper about "dishonesty in behavioral economics experiments" 
        matches 3 primary keywords → high relevance score with bonus.
        """)
        
        # Visual relevance threshold
        st.write("")
        st.write("**Your current threshold:**")
        
        threshold = config["filters"]["min_relevance"]
        if threshold <= 0.1:
            st.success(f"🟢 {threshold:.0%} - Very inclusive (more papers, some may be tangential)")
        elif threshold <= 0.2:
            st.info(f"🔵 {threshold:.0%} - Balanced (good mix of relevance and volume)")
        elif threshold <= 0.3:
            st.warning(f"🟡 {threshold:.0%} - Selective (fewer but more relevant papers)")
        else:
            st.error(f"🔴 {threshold:.0%} - Very strict (may miss relevant papers)")
    
    # =====================
    # TAB 5: Recent Papers
    # =====================
    with tab5:
        st.subheader("📊 Papers from Last Digest")
        
        papers = load_latest_papers()
        
        if not papers:
            st.info("No papers yet. Run the digest to see results here.")
        else:
            st.success(f"**{len(papers)} verified papers** in last digest")
            
            for paper in papers:
                with st.expander(f"📄 {paper.get('title', 'Untitled')[:80]}..."):
                    st.write(f"**Authors:** {', '.join(paper.get('authors', [])[:3])}")
                    st.write(f"**Source:** {paper.get('source', 'Unknown')}")
                    st.write(f"**DOI:** [{paper.get('doi', 'N/A')}]({paper.get('url', '#')})")
                    st.write(f"**Relevance:** {paper.get('relevance_score', 0):.0%}")
                    st.write(f"**Date:** {paper.get('published_date', 'Unknown')}")
                    
                    if paper.get('summary'):
                        st.write("---")
                        st.write(f"**Summary:** {paper.get('summary')}")
                    
                    if paper.get('research_connection'):
                        st.info(f"🔗 **Research Connection:** {paper.get('research_connection')}")
    
    # Store config in session state
    st.session_state['config'] = config
    
    # Footer
    st.divider()
    st.caption("Eugen's Research Digest • All papers verified against Crossref DOI registry")


if __name__ == "__main__":
    main()
