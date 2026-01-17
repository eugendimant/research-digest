#!/usr/bin/env python3
"""
Eugen's Weekly Research Digest - Production Version
With REAL DOI verification - every paper is validated before inclusion.
NO fake/test papers. All papers are fetched from real APIs and verified.
"""

import os
import sys
import json
import re
import hashlib
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import urllib.request
import urllib.parse
import urllib.error
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "email": {
        "recipient": "edimant@sas.upenn.edu",
        "sender_name": "Eugen's Research Digest"
    },
    "researcher_profile": {
        "name": "Eugen Dimant",
        "affiliation": "University of Pennsylvania / Stanford University",
        "focus_areas": [
            "dishonesty and cheating behavior",
            "social norms and norm enforcement",
            "corruption and unethical behavior",
            "behavioral interventions and nudges",
            "experimental and field methods"
        ]
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
            "cryptocurrency", "bitcoin", "ethereum", "blockchain",
            "stock prediction", "algorithmic trading", "sports betting"
        ]
    },
    "filters": {
        "days_lookback": 14,  # Increased for more results
        "min_relevance": 0.12,
        "max_papers": 20,
        "skip_seen_papers": True
    }
}

# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    doi: str
    source: str
    published_date: str
    relevance_score: float = 0.0
    summary: str = ""
    research_connection: str = ""
    verified: bool = False
    
    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""
    
    @property
    def paper_id(self) -> str:
        return hashlib.md5(self.doi.encode()).hexdigest()[:12] if self.doi else ""
    
    def to_dict(self):
        d = asdict(self)
        d['url'] = self.url
        d['paper_id'] = self.paper_id
        return d

# =============================================================================
# DOI VERIFICATION - CRITICAL FOR PREVENTING FAKE PAPERS
# =============================================================================

# Cache for verified DOIs to avoid repeated checks
VERIFIED_DOIS_CACHE: Dict[str, bool] = {}

def verify_doi_exists(doi: str) -> Tuple[bool, Optional[dict]]:
    """
    Verify a DOI actually exists by querying Crossref API.
    Returns (exists, metadata) where metadata contains real paper info.
    This is the CRITICAL function that prevents fake papers.
    """
    if not doi:
        return False, None
    
    # Clean DOI
    doi = doi.strip()
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
    
    # Check cache first
    if doi in VERIFIED_DOIS_CACHE:
        if not VERIFIED_DOIS_CACHE[doi]:
            return False, None
    
    # Validate DOI format
    if not re.match(r'^10\.\d{4,}/[^\s]+$', doi):
        VERIFIED_DOIS_CACHE[doi] = False
        return False, None
    
    # Query Crossref to verify DOI exists and get metadata
    try:
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
        headers = {
            'User-Agent': 'EugenResearchDigest/2.0 (mailto:edimant@sas.upenn.edu)',
            'Accept': 'application/json'
        }
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        if data.get('status') == 'ok' and data.get('message'):
            VERIFIED_DOIS_CACHE[doi] = True
            return True, data['message']
        
        VERIFIED_DOIS_CACHE[doi] = False
        return False, None
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"DOI not found: {doi}")
        VERIFIED_DOIS_CACHE[doi] = False
        return False, None
    except Exception as e:
        logger.debug(f"DOI verification error for {doi}: {e}")
        # Don't cache network errors
        return False, None


def extract_verified_paper_from_crossref(crossref_data: dict, doi: str, source: str) -> Optional[Paper]:
    """
    Extract paper information from verified Crossref response.
    Only returns a Paper if we have confirmed real metadata.
    """
    if not crossref_data:
        return None
    
    # Get title
    title_list = crossref_data.get("title", [])
    title = title_list[0] if title_list else ""
    if not title:
        return None
    
    # Get abstract
    abstract = crossref_data.get("abstract", "")
    if abstract:
        abstract = re.sub(r'<[^>]+>', '', abstract)  # Remove HTML
        abstract = ' '.join(abstract.split())  # Normalize whitespace
    
    # Get authors
    authors = []
    for author in crossref_data.get("author", [])[:5]:
        given = author.get("given", "")
        family = author.get("family", "")
        if family:
            name = f"{given} {family}".strip() if given else family
            authors.append(name)
    
    # Get publication date
    pub_date = crossref_data.get("published", {}).get("date-parts", [[]])
    if pub_date and pub_date[0]:
        parts = pub_date[0]
        year = parts[0] if len(parts) > 0 else datetime.now().year
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        pub_date_str = f"{year}-{month:02d}-{day:02d}"
    else:
        # Try other date fields
        created = crossref_data.get("created", {}).get("date-parts", [[]])
        if created and created[0]:
            parts = created[0]
            year = parts[0] if len(parts) > 0 else datetime.now().year
            month = parts[1] if len(parts) > 1 else 1
            day = parts[2] if len(parts) > 2 else 1
            pub_date_str = f"{year}-{month:02d}-{day:02d}"
        else:
            pub_date_str = datetime.now().strftime("%Y-%m-%d")
    
    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        doi=doi,
        source=source,
        published_date=pub_date_str,
        verified=True
    )

# =============================================================================
# SEEN PAPERS TRACKING
# =============================================================================

SEEN_PAPERS_FILE = "seen_papers.json"

def load_seen_papers() -> Set[str]:
    try:
        with open(SEEN_PAPERS_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("paper_ids", []))
    except FileNotFoundError:
        return set()

def save_seen_papers(paper_ids: Set[str]):
    ids_list = list(paper_ids)[-2000:]
    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump({
            "paper_ids": ids_list,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)

# =============================================================================
# RELEVANCE SCORING
# =============================================================================

def calculate_relevance(title: str, abstract: str) -> float:
    text = f"{title} {abstract}".lower()
    topics = CONFIG["topics"]
    
    primary = topics.get("primary", [])
    secondary = topics.get("secondary", [])
    
    primary_matches = sum(1 for kw in primary if kw.lower() in text)
    secondary_matches = sum(1 for kw in secondary if kw.lower() in text)
    
    score = (0.65 * (primary_matches / max(len(primary), 1)) + 
             0.35 * (secondary_matches / max(len(secondary), 1)))
    
    if primary_matches >= 2:
        score = min(score * 1.3, 1.0)
    
    return round(score, 3)

def should_exclude(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    return any(term.lower() in text for term in CONFIG["topics"].get("exclude", []))

# =============================================================================
# HTTP HELPERS
# =============================================================================

def http_get_json(url: str, headers: dict = None, timeout: int = 20) -> Optional[dict]:
    if headers is None:
        headers = {'User-Agent': 'EugenResearchDigest/2.0 (mailto:edimant@sas.upenn.edu)'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        logger.debug(f"HTTP error: {e}")
        return None

# =============================================================================
# PAPER FETCHING - CROSSREF (Most reliable, direct DOI source)
# =============================================================================

def fetch_from_crossref() -> List[Paper]:
    """
    Fetch papers directly from Crossref - the authoritative DOI registry.
    Every paper from here has a verified, real DOI.
    """
    logger.info("Fetching from Crossref (authoritative DOI source)...")
    papers = []
    
    days_back = CONFIG["filters"]["days_lookback"]
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    keywords = CONFIG["topics"]["primary"][:6]
    
    for keyword in keywords:
        try:
            url = "https://api.crossref.org/works"
            params = {
                "query": keyword,
                "filter": f"from-pub-date:{from_date},type:journal-article",
                "sort": "published",
                "order": "desc",
                "rows": 25,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data or "message" not in data:
                continue
            
            for item in data["message"].get("items", []):
                doi = item.get("DOI", "")
                if not doi:
                    continue
                
                # Extract paper from Crossref data (already verified since it's from Crossref)
                paper = extract_verified_paper_from_crossref(item, doi, "Crossref")
                if not paper:
                    continue
                
                if should_exclude(paper.title, paper.abstract):
                    continue
                
                paper.relevance_score = calculate_relevance(paper.title, paper.abstract)
                
                if paper.relevance_score >= 0.08:
                    papers.append(paper)
            
            time.sleep(0.15)
            
        except Exception as e:
            logger.error(f"Crossref error for '{keyword}': {e}")
    
    # Deduplicate by DOI
    seen = set()
    unique = [p for p in papers if p.doi not in seen and not seen.add(p.doi)]
    
    logger.info(f"Crossref: {len(unique)} verified papers")
    return unique

# =============================================================================
# PAPER FETCHING - OPENALEX (with DOI verification)
# =============================================================================

def fetch_from_openalex() -> List[Paper]:
    """
    Fetch from OpenAlex, then verify each DOI against Crossref.
    """
    logger.info("Fetching from OpenAlex...")
    papers = []
    
    days_back = CONFIG["filters"]["days_lookback"]
    cutoff_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    keywords = CONFIG["topics"]["primary"][:5]
    
    for keyword in keywords:
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": keyword,
                "filter": f"from_publication_date:{cutoff_str},has_doi:true,type:article",
                "sort": "publication_date:desc",
                "per_page": 20,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data:
                continue
            
            for work in data.get("results", []):
                doi_url = work.get("doi", "")
                if not doi_url:
                    continue
                
                doi = doi_url.replace("https://doi.org/", "")
                
                # VERIFY the DOI actually exists
                exists, crossref_data = verify_doi_exists(doi)
                if not exists:
                    logger.debug(f"Skipping unverified DOI: {doi}")
                    continue
                
                # Use Crossref data for accuracy (it's the authoritative source)
                paper = extract_verified_paper_from_crossref(crossref_data, doi, "OpenAlex")
                if not paper:
                    continue
                
                if should_exclude(paper.title, paper.abstract):
                    continue
                
                paper.relevance_score = calculate_relevance(paper.title, paper.abstract)
                
                if paper.relevance_score >= 0.08:
                    papers.append(paper)
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"OpenAlex error for '{keyword}': {e}")
    
    seen = set()
    unique = [p for p in papers if p.doi not in seen and not seen.add(p.doi)]
    
    logger.info(f"OpenAlex: {len(unique)} verified papers")
    return unique

# =============================================================================
# PAPER FETCHING - SEMANTIC SCHOLAR (with DOI verification)
# =============================================================================

def fetch_from_semantic_scholar() -> List[Paper]:
    """
    Fetch from Semantic Scholar, then verify each DOI.
    """
    logger.info("Fetching from Semantic Scholar...")
    papers = []
    
    keywords = CONFIG["topics"]["primary"][:4]
    
    for keyword in keywords:
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": keyword,
                "limit": 20,
                "fields": "title,authors,abstract,externalIds,year,publicationDate"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data:
                continue
            
            for work in data.get("data", []):
                ext_ids = work.get("externalIds", {})
                doi = ext_ids.get("DOI", "")
                
                if not doi:
                    continue
                
                # VERIFY the DOI
                exists, crossref_data = verify_doi_exists(doi)
                if not exists:
                    continue
                
                paper = extract_verified_paper_from_crossref(crossref_data, doi, "Semantic Scholar")
                if not paper:
                    continue
                
                if should_exclude(paper.title, paper.abstract):
                    continue
                
                paper.relevance_score = calculate_relevance(paper.title, paper.abstract)
                
                if paper.relevance_score >= 0.08:
                    papers.append(paper)
            
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Semantic Scholar error for '{keyword}': {e}")
    
    seen = set()
    unique = [p for p in papers if p.doi not in seen and not seen.add(p.doi)]
    
    logger.info(f"Semantic Scholar: {len(unique)} verified papers")
    return unique

# =============================================================================
# PAPER FETCHING - PUBMED (with DOI verification)
# =============================================================================

def fetch_from_pubmed() -> List[Paper]:
    """
    Fetch from PubMed, then verify each DOI.
    """
    logger.info("Fetching from PubMed...")
    papers = []
    
    days_back = CONFIG["filters"]["days_lookback"]
    keywords = CONFIG["topics"]["primary"][:3]
    query = " OR ".join(f'"{kw}"' for kw in keywords)
    
    try:
        # Search
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"({query}) AND (\"last {days_back} days\"[dp])",
            "retmax": 40,
            "retmode": "json"
        }
        
        search_data = http_get_json(f"{search_url}?{urllib.parse.urlencode(search_params)}")
        if not search_data:
            return []
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
        
        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list[:25]),
            "retmode": "json"
        }
        
        detail_data = http_get_json(f"{fetch_url}?{urllib.parse.urlencode(fetch_params)}")
        if not detail_data:
            return []
        
        results = detail_data.get("result", {})
        
        for pmid in id_list[:25]:
            article = results.get(pmid, {})
            if not article or not isinstance(article, dict):
                continue
            
            # Get DOI from article IDs
            doi = ""
            for aid in article.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
                    break
            
            if not doi:
                continue
            
            # VERIFY the DOI
            exists, crossref_data = verify_doi_exists(doi)
            if not exists:
                continue
            
            paper = extract_verified_paper_from_crossref(crossref_data, doi, "PubMed")
            if not paper:
                continue
            
            if should_exclude(paper.title, paper.abstract):
                continue
            
            paper.relevance_score = calculate_relevance(paper.title, paper.abstract)
            
            if paper.relevance_score >= 0.08:
                papers.append(paper)
        
    except Exception as e:
        logger.error(f"PubMed error: {e}")
    
    seen = set()
    unique = [p for p in papers if p.doi not in seen and not seen.add(p.doi)]
    
    logger.info(f"PubMed: {len(unique)} verified papers")
    return unique

# =============================================================================
# FETCH ALL AND DEDUPLICATE
# =============================================================================

def fetch_all_papers() -> List[Paper]:
    """
    Fetch from all sources. Every paper is verified to have a real, working DOI.
    """
    all_papers = []
    
    # Crossref first (most authoritative)
    all_papers.extend(fetch_from_crossref())
    
    # Other sources (all with DOI verification)
    all_papers.extend(fetch_from_openalex())
    all_papers.extend(fetch_from_semantic_scholar())
    all_papers.extend(fetch_from_pubmed())
    
    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for p in all_papers:
        if p.doi and p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    # Filter by relevance
    min_rel = CONFIG["filters"]["min_relevance"]
    filtered = [p for p in unique if p.relevance_score >= min_rel]
    
    # Sort by relevance
    filtered.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # Filter seen papers
    if CONFIG["filters"]["skip_seen_papers"]:
        seen_ids = load_seen_papers()
        filtered = [p for p in filtered if p.paper_id not in seen_ids]
    
    # Limit
    max_papers = CONFIG["filters"]["max_papers"]
    final = filtered[:max_papers]
    
    # Final verification check
    final = [p for p in final if p.verified and p.doi]
    
    logger.info(f"Pipeline: {len(all_papers)} total → {len(unique)} unique → {len(final)} final (all verified)")
    return final

# =============================================================================
# AI SUMMARIES (only for verified papers)
# =============================================================================

def call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    except:
        return ""


def generate_summary_and_connection(paper: Paper) -> Tuple[str, str]:
    """Generate summary and research connection for a verified paper."""
    if not paper.abstract:
        return "", ""
    
    focus = ", ".join(CONFIG["researcher_profile"]["focus_areas"][:3])
    
    prompt = f"""Paper for Professor Eugen Dimant (behavioral economist studying {focus}):

Title: {paper.title}
Abstract: {paper.abstract[:500]}

Provide:
1. SUMMARY (2 sentences): Main finding and method
2. CONNECTION (2 sentences): How this connects to Eugen's research

Format exactly as:
SUMMARY: [text]
CONNECTION: [text]"""

    response = call_gemini(prompt)
    
    summary = ""
    connection = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.upper().startswith("CONNECTION:"):
            connection = line[11:].strip()
    
    if not summary and paper.abstract:
        summary = paper.abstract[:200] + "..."
    
    return summary, connection


def generate_highlights(papers: List[Paper]) -> str:
    if not papers:
        return "No new verified papers found this week."
    
    titles = "\n".join([f"• {p.title}" for p in papers[:8]])
    
    prompt = f"""Based on these behavioral economics papers, write 2-3 thematic highlights (use • bullets, **bold** themes):

{titles}

Focus on themes related to dishonesty, social norms, and experimental methods. Each highlight: 1-2 sentences."""

    result = call_gemini(prompt)
    return result if result else "• **New Research**: Several verified papers published this week in behavioral economics."

# =============================================================================
# CATEGORIZATION
# =============================================================================

def categorize_papers(papers: List[Paper]) -> Dict[str, List[Paper]]:
    categories = {
        "Dishonesty & Ethics": [],
        "Social Norms & Cooperation": [],
        "Experimental Methods": [],
        "Policy & Interventions": [],
        "Other Relevant": []
    }
    
    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        
        if any(kw in text for kw in ["dishonest", "cheat", "corrupt", "fraud", "lying", "deception", "honesty"]):
            categories["Dishonesty & Ethics"].append(paper)
        elif any(kw in text for kw in ["norm", "cooperat", "trust", "prosocial", "public good"]):
            categories["Social Norms & Cooperation"].append(paper)
        elif any(kw in text for kw in ["experiment", "rct", "randomized", "treatment effect"]):
            categories["Experimental Methods"].append(paper)
        elif any(kw in text for kw in ["policy", "intervention", "nudge", "regulation"]):
            categories["Policy & Interventions"].append(paper)
        else:
            categories["Other Relevant"].append(paper)
    
    return {k: v for k, v in categories.items() if v}

# =============================================================================
# EMAIL GENERATION
# =============================================================================

def format_paper_html(paper: Paper) -> str:
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."
    
    badge = ""
    if paper.relevance_score >= 0.4:
        badge = '<span style="background:#059669;color:white;padding:2px 8px;border-radius:12px;font-size:10px;margin-left:8px;">High Match</span>'
    elif paper.relevance_score >= 0.25:
        badge = '<span style="background:#d97706;color:white;padding:2px 8px;border-radius:12px;font-size:10px;margin-left:8px;">Good Match</span>'
    
    summary = paper.summary if paper.summary else (paper.abstract[:250] + "..." if paper.abstract else "No abstract available.")
    
    connection_html = ""
    if paper.research_connection:
        connection_html = f'''
        <div style="margin-top:10px;padding:10px;background:#fef3c7;border-radius:6px;border-left:3px solid #f59e0b;">
            <p style="margin:0;font-size:12px;color:#92400e;">
                <strong>🔗 Research Connection:</strong> {paper.research_connection}
            </p>
        </div>'''
    
    return f'''
    <div style="margin-bottom:24px;padding:18px;background:#f8fafc;border-radius:10px;border-left:4px solid #3b82f6;">
        <h3 style="margin:0 0 10px 0;font-size:15px;line-height:1.4;">
            <a href="{paper.url}" style="color:#1e40af;text-decoration:none;" target="_blank">{paper.title}</a>
            {badge}
        </h3>
        <p style="margin:0 0 8px 0;color:#64748b;font-size:12px;">
            {authors_str} • <span style="background:#e2e8f0;padding:2px 6px;border-radius:4px;">{paper.source}</span> • {paper.published_date}
        </p>
        <p style="margin:0 0 6px 0;color:#64748b;font-size:11px;">
            <strong>DOI:</strong> <a href="{paper.url}" style="color:#3b82f6;" target="_blank">{paper.doi}</a> ✓ Verified
        </p>
        <p style="margin:0;color:#475569;font-size:13px;line-height:1.5;">{summary}</p>
        {connection_html}
    </div>'''


def generate_email_html(papers: List[Paper], highlights: str, categories: Dict[str, List[Paper]]) -> str:
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # Highlights
    lines = highlights.strip().split('\n')
    highlight_items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[•\-\*]\s*', '', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        highlight_items.append(f'<li style="margin-bottom:10px;line-height:1.5;color:#1e3a5f;">{line}</li>')
    highlights_html = f'<ul style="margin:0;padding-left:20px;">{"".join(highlight_items)}</ul>'
    
    # Papers by category
    papers_html = ""
    if categories:
        for cat_name, cat_papers in categories.items():
            cat_html = "".join([format_paper_html(p) for p in cat_papers])
            papers_html += f'''
            <div style="margin-bottom:32px;">
                <h2 style="margin:0 0 18px 0;font-size:18px;color:#1e293b;border-bottom:2px solid #3b82f6;padding-bottom:8px;">
                    {cat_name} <span style="color:#64748b;font-weight:normal;font-size:14px;">({len(cat_papers)})</span>
                </h2>
                {cat_html}
            </div>'''
    else:
        papers_html = "".join([format_paper_html(p) for p in papers])
    
    sources = sorted(set(p.source for p in papers)) if papers else []
    
    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;margin:0;padding:20px;">
    <div style="max-width:720px;margin:0 auto;background:white;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.12);">
        
        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);padding:40px 32px;text-align:center;">
            <h1 style="margin:0;color:white;font-size:28px;font-weight:700;">Eugen's Weekly<br>Research Digest</h1>
            <p style="margin:14px 0 0 0;color:#93c5fd;font-size:15px;">{date_str}</p>
        </div>
        
        <div style="margin:28px;padding:24px;background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border-radius:12px;border:1px solid #bfdbfe;">
            <h2 style="margin:0 0 14px 0;font-size:18px;color:#1e40af;">📊 This Week's Highlights</h2>
            <p style="margin:0 0 16px 0;color:#1e40af;font-size:14px;">
                <strong>{len(papers)} verified papers</strong> from {', '.join(sources) if sources else 'academic databases'}
            </p>
            {highlights_html}
        </div>
        
        <div style="margin:0 28px 28px;padding:14px 18px;background:#f0fdf4;border-radius:8px;border-left:4px solid #22c55e;">
            <p style="margin:0;font-size:12px;color:#166534;">
                ✓ <strong>All papers verified</strong> - DOIs checked against Crossref registry. Click any link to access the real paper.
            </p>
        </div>
        
        <div style="padding:0 28px 28px;">
            {papers_html if papers else '<p style="color:#64748b;text-align:center;">No new papers matching your criteria this week.</p>'}
        </div>
        
        <div style="background:#f8fafc;padding:24px 28px;text-align:center;border-top:1px solid #e2e8f0;">
            <p style="margin:0 0 8px 0;color:#64748b;font-size:12px;">
                All papers verified against Crossref DOI registry • Only new papers since last digest
            </p>
            <p style="margin:0;color:#94a3b8;font-size:11px;">
                Sources: Crossref, OpenAlex, Semantic Scholar, PubMed
            </p>
        </div>
    </div>
</body>
</html>'''

# =============================================================================
# EMAIL SENDING
# =============================================================================

def send_email(html: str) -> bool:
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    recipient = CONFIG["email"]["recipient"]
    
    if not all([smtp_user, smtp_password, recipient]):
        logger.error("Missing email configuration")
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Eugen's Research Digest - {datetime.now().strftime('%B %d, %Y')}"
        msg["From"] = f"Eugen's Research Digest <{smtp_user}>"
        msg["To"] = recipient
        
        msg.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"✓ Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    preview_mode = "--preview" in sys.argv
    
    logger.info("=" * 60)
    logger.info("EUGEN'S RESEARCH DIGEST - VERIFIED PAPERS ONLY")
    logger.info("=" * 60)
    
    # ALWAYS fetch real papers - no fake test data
    logger.info("Fetching and verifying papers from academic databases...")
    papers = fetch_all_papers()
    
    if papers:
        logger.info(f"Generating AI summaries for {len(papers)} verified papers...")
        highlights = generate_highlights(papers)
        
        for i, paper in enumerate(papers):
            logger.info(f"  [{i+1}/{len(papers)}] {paper.title[:50]}...")
            summary, connection = generate_summary_and_connection(paper)
            paper.summary = summary
            paper.research_connection = connection
            time.sleep(0.2)
    else:
        highlights = "No new papers found matching your criteria this week."
    
    categories = categorize_papers(papers)
    
    html = generate_email_html(papers, highlights, categories)
    
    # Save outputs
    with open("latest_digest.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    with open("latest_papers.json", "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in papers], f, indent=2)
    
    logger.info(f"\n📊 Final: {len(papers)} verified papers")
    for p in papers:
        logger.info(f"  ✓ {p.doi} - {p.title[:50]}...")
    
    if preview_mode:
        logger.info("\n✓ Preview mode - no email sent")
        return True
    
    success = send_email(html)
    
    if success:
        seen = load_seen_papers()
        for p in papers:
            seen.add(p.paper_id)
        save_seen_papers(seen)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
