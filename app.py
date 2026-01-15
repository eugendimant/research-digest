"""
Eugen's Research Digest - Configuration Interface
A Streamlit app for managing digest settings.
"""

import streamlit as st
import yaml
from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Eugen's Research Digest Settings",
    page_icon="📚",
    layout="wide"
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
    }
    .source-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .success-box {
        background: #d1fae5;
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIG HANDLING
# =============================================================================

CONFIG_FILE = "config.yaml"
SEEN_PAPERS_FILE = "seen_papers.json"

def load_config():
    """Load config from YAML file."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return get_default_config()

def save_config(config):
    """Save config to YAML file."""
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def get_default_config():
    """Return default configuration."""
    return {
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
            "primary": ["behavioral economics", "dishonesty", "social norms"],
            "secondary": ["nudge", "cooperation", "field experiment"],
            "exclude": ["cryptocurrency", "bitcoin"]
        },
        "sources": {
            "arxiv": {"enabled": True},
            "openalex": {"enabled": True},
            "semantic_scholar": {"enabled": True},
            "pubmed": {"enabled": True},
            "nber": {"enabled": True},
            "repec": {"enabled": True}
        },
        "filters": {
            "days_lookback": 7,
            "min_relevance": 0.20,
            "max_papers": 20,
            "skip_seen_papers": True
        },
        "display": {
            "show_relevance_badges": True,
            "group_by_category": True,
            "ai_summaries": True,
            "show_opportunities": True
        }
    }

def load_seen_papers():
    """Load seen papers data."""
    try:
        with open(SEEN_PAPERS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"paper_ids": [], "last_updated": None}

def clear_seen_papers():
    """Clear the seen papers list."""
    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump({"paper_ids": [], "last_updated": datetime.now().isoformat()}, f)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Eugen's Research Digest</h1>
        <p>Configure your automated research paper alerts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load current config
    config = load_config()
    
    # Sidebar for quick actions
    with st.sidebar:
        st.header("⚡ Quick Actions")
        
        if st.button("💾 Save All Changes", use_container_width=True):
            save_config(st.session_state.get('config', config))
            st.success("✅ Configuration saved!")
        
        st.divider()
        
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            config = get_default_config()
            save_config(config)
            st.success("Reset complete!")
            st.rerun()
        
        st.divider()
        
        seen_data = load_seen_papers()
        st.metric("Papers Tracked", len(seen_data.get("paper_ids", [])))
        
        if st.button("🗑️ Clear Paper History", use_container_width=True):
            clear_seen_papers()
            st.success("History cleared - next digest will include all papers")
        
        st.divider()
        st.caption("Last updated: " + (seen_data.get("last_updated", "Never")[:10] if seen_data.get("last_updated") else "Never"))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📧 Email & Schedule", 
        "🔍 Research Topics", 
        "📚 Sources",
        "⚙️ Filters",
        "🎨 Display"
    ])
    
    # ===================
    # TAB 1: Email & Schedule
    # ===================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📧 Email Settings")
            
            config["email"]["recipient"] = st.text_input(
                "Recipient Email",
                value=config.get("email", {}).get("recipient", ""),
                help="Where to send the digest"
            )
            
            config["email"]["sender_name"] = st.text_input(
                "Sender Name",
                value=config.get("email", {}).get("sender_name", "Eugen's Research Digest"),
                help="Name shown in the 'From' field"
            )
        
        with col2:
            st.subheader("📅 Schedule")
            
            frequency = st.selectbox(
                "Frequency",
                options=["daily", "weekly", "biweekly", "monthly"],
                index=["daily", "weekly", "biweekly", "monthly"].index(
                    config.get("schedule", {}).get("frequency", "weekly")
                ),
                help="How often to receive the digest"
            )
            config["schedule"]["frequency"] = frequency
            
            if frequency in ["weekly", "biweekly"]:
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_idx = st.selectbox(
                    "Day of Week",
                    options=range(7),
                    format_func=lambda x: day_names[x],
                    index=config.get("schedule", {}).get("day_of_week", 6),
                    help="Which day to send the digest"
                )
                config["schedule"]["day_of_week"] = day_idx
            
            hour = st.slider(
                "Time (UTC)",
                min_value=0,
                max_value=23,
                value=config.get("schedule", {}).get("hour_utc", 23),
                help="23 UTC = 6 PM EST"
            )
            config["schedule"]["hour_utc"] = hour
            
            # Show local time conversion
            est_hour = (hour - 5) % 24
            st.caption(f"📍 This is {est_hour}:00 EST / {(hour - 8) % 24}:00 PST")
    
    # ===================
    # TAB 2: Research Topics
    # ===================
    with tab2:
        st.subheader("🎯 Primary Keywords")
        st.caption("High priority - papers with these terms are ranked highest")
        
        primary_keywords = config.get("topics", {}).get("primary", [])
        primary_text = st.text_area(
            "Primary Keywords (one per line)",
            value="\n".join(primary_keywords),
            height=200,
            help="These keywords have the highest weight in relevance scoring"
        )
        config["topics"]["primary"] = [k.strip() for k in primary_text.split("\n") if k.strip()]
        
        st.divider()
        
        st.subheader("📌 Secondary Keywords")
        st.caption("Medium priority - boost papers that also contain these")
        
        secondary_keywords = config.get("topics", {}).get("secondary", [])
        secondary_text = st.text_area(
            "Secondary Keywords (one per line)",
            value="\n".join(secondary_keywords),
            height=150,
            help="These keywords have medium weight"
        )
        config["topics"]["secondary"] = [k.strip() for k in secondary_text.split("\n") if k.strip()]
        
        st.divider()
        
        st.subheader("🚫 Exclude Terms")
        st.caption("Papers containing these terms will be filtered out")
        
        exclude_terms = config.get("topics", {}).get("exclude", [])
        exclude_text = st.text_area(
            "Exclude Terms (one per line)",
            value="\n".join(exclude_terms),
            height=100,
            help="Papers with these terms are excluded"
        )
        config["topics"]["exclude"] = [k.strip() for k in exclude_text.split("\n") if k.strip()]
    
    # ===================
    # TAB 3: Sources
    # ===================
    with tab3:
        st.subheader("📚 Paper Sources")
        st.caption("Enable or disable different academic databases")
        
        sources_info = {
            "arxiv": {
                "name": "arXiv",
                "description": "Preprints in economics, game theory, and computational social science",
                "icon": "📄"
            },
            "openalex": {
                "name": "OpenAlex", 
                "description": "Comprehensive database of 250M+ academic works",
                "icon": "🌐"
            },
            "semantic_scholar": {
                "name": "Semantic Scholar",
                "description": "AI-powered academic search from Allen Institute",
                "icon": "🤖"
            },
            "pubmed": {
                "name": "PubMed",
                "description": "Biomedical and psychology literature",
                "icon": "🏥"
            },
            "nber": {
                "name": "NBER",
                "description": "National Bureau of Economic Research working papers",
                "icon": "📊"
            },
            "repec": {
                "name": "RePEc",
                "description": "Research Papers in Economics",
                "icon": "📈"
            }
        }
        
        col1, col2 = st.columns(2)
        
        for i, (source_key, source_info) in enumerate(sources_info.items()):
            col = col1 if i % 2 == 0 else col2
            
            with col:
                current_enabled = config.get("sources", {}).get(source_key, {}).get("enabled", True)
                
                enabled = st.checkbox(
                    f"{source_info['icon']} **{source_info['name']}**",
                    value=current_enabled,
                    help=source_info['description'],
                    key=f"source_{source_key}"
                )
                
                if source_key not in config.get("sources", {}):
                    config["sources"][source_key] = {}
                config["sources"][source_key]["enabled"] = enabled
                
                st.caption(source_info['description'])
                st.write("")
    
    # ===================
    # TAB 4: Filters
    # ===================
    with tab4:
        st.subheader("🔧 Filter Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            days = st.slider(
                "Days Lookback",
                min_value=1,
                max_value=30,
                value=config.get("filters", {}).get("days_lookback", 7),
                help="Only include papers from the last N days"
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
                value=config.get("filters", {}).get("min_relevance", 0.20),
                step=0.05,
                help="Papers below this score are filtered out (0 = include all)"
            )
            config["filters"]["min_relevance"] = min_rel
            
            skip_seen = st.checkbox(
                "Skip Previously Sent Papers",
                value=config.get("filters", {}).get("skip_seen_papers", True),
                help="Only show papers you haven't received before"
            )
            config["filters"]["skip_seen_papers"] = skip_seen
        
        # Relevance explanation
        st.divider()
        st.subheader("📊 How Relevance Scoring Works")
        st.markdown("""
        Each paper receives a relevance score from 0 to 1 based on keyword matches:
        
        - **Primary keywords**: 65% weight - matches here have the biggest impact
        - **Secondary keywords**: 35% weight - additional relevant terms
        - **Bonus**: Papers matching 2+ primary keywords get a 20% boost
        
        Example: A paper about "dishonesty in behavioral economics experiments" would score high 
        because it matches multiple primary keywords.
        """)
    
    # ===================
    # TAB 5: Display
    # ===================
    with tab5:
        st.subheader("🎨 Display Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_badges = st.checkbox(
                "Show Relevance Badges",
                value=config.get("display", {}).get("show_relevance_badges", True),
                help="Show 'High Match' and 'Good Match' labels on papers"
            )
            config["display"]["show_relevance_badges"] = show_badges
            
            group_by_cat = st.checkbox(
                "Group by Category",
                value=config.get("display", {}).get("group_by_category", True),
                help="Organize papers into categories like 'Dishonesty & Ethics'"
            )
            config["display"]["group_by_category"] = group_by_cat
        
        with col2:
            ai_summaries = st.checkbox(
                "AI-Generated Summaries",
                value=config.get("display", {}).get("ai_summaries", True),
                help="Use Gemini to create paper summaries (requires API key)"
            )
            config["display"]["ai_summaries"] = ai_summaries
            
            show_opps = st.checkbox(
                "Show Research Opportunities",
                value=config.get("display", {}).get("show_opportunities", True),
                help="Include AI-suggested research ideas"
            )
            config["display"]["show_opportunities"] = show_opps
    
    # Store config in session state
    st.session_state['config'] = config
    
    # Bottom save button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True, type="primary"):
            save_config(config)
            st.success("✅ Configuration saved successfully!")
            st.balloons()


if __name__ == "__main__":
    main()
