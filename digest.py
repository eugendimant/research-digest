#!/usr/bin/env python3
"""
Eugen's Weekly Research Digest
Comprehensive paper fetching, AI summaries, and email delivery.
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
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field, asdict
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import yaml
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    published_date: str
    relevance_score: float = 0.0
    summary: str = ""
    paper_id: str = ""
    
    def __post_init__(self):
        if not self.paper_id:
            # Generate unique ID from title
            self.paper_id = hashlib.md5(self.title.lower().encode()).hexdigest()[:12]
    
    def to_dict(self):
        return asdict(self)

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        sys.exit(1)

# =============================================================================
# SEEN PAPERS TRACKING
# =============================================================================

SEEN_PAPERS_FILE = "seen_papers.json"

def load_seen_papers() -> Set[str]:
    """Load IDs of previously sent papers."""
    try:
        with open(SEEN_PAPERS_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("paper_ids", []))
    except FileNotFoundError:
        return set()

def save_seen_papers(paper_ids: Set[str]):
    """Save IDs of sent papers."""
    # Keep last 1000 to prevent file from growing too large
    ids_list = list(paper_ids)[-1000:]
    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump({
            "paper_ids": ids_list,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Saved {len(ids_list)} seen paper IDs")

# =============================================================================
# RELEVANCE SCORING
# =============================================================================

def calculate_relevance(title: str, abstract: str, topics: dict) -> float:
    """Calculate relevance score based on keyword matches."""
    text = f"{title} {abstract}".lower()
    
    primary = topics.get("primary", [])
    secondary = topics.get("secondary", [])
    
    # Count matches
    primary_matches = sum(1 for kw in primary if kw.lower() in text)
    secondary_matches = sum(1 for kw in secondary if kw.lower() in text)
    
    # Weighted scoring
    total_primary = max(len(primary), 1)
    total_secondary = max(len(secondary), 1)
    
    score = (0.65 * (primary_matches / total_primary) + 
             0.35 * (secondary_matches / total_secondary))
    
    # Boost for multiple primary matches
    if primary_matches >= 2:
        score = min(score * 1.2, 1.0)
    
    return round(score, 3)

def should_exclude(title: str, abstract: str, exclude_terms: list) -> bool:
    """Check if paper should be excluded."""
    text = f"{title} {abstract}".lower()
    return any(term.lower() in text for term in exclude_terms)

# =============================================================================
# PAPER FETCHING - ARXIV
# =============================================================================

def fetch_arxiv(config: dict) -> List[Paper]:
    """Fetch papers from arxiv."""
    if not config.get("sources", {}).get("arxiv", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from arXiv...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    categories = config.get("sources", {}).get("arxiv", {}).get("categories", ["econ.GN"])
    keywords = topics.get("primary", [])[:5]
    
    # Build query
    search_terms = " OR ".join(f'all:"{kw}"' for kw in keywords)
    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    query = f"({search_terms}) AND ({cat_query})"
    
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 100,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode('utf-8')
        
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            if title_elem is None:
                continue
            title = title_elem.text.strip().replace('\n', ' ')
            
            abstract_elem = entry.find('atom:summary', ns)
            abstract = abstract_elem.text.strip().replace('\n', ' ') if abstract_elem is not None else ""
            
            published_elem = entry.find('atom:published', ns)
            if published_elem is not None:
                pub_date = datetime.strptime(published_elem.text[:10], "%Y-%m-%d")
                if pub_date < cutoff_date:
                    continue
                pub_date_str = pub_date.strftime("%Y-%m-%d")
            else:
                pub_date_str = datetime.now().strftime("%Y-%m-%d")
            
            if should_exclude(title, abstract, topics.get("exclude", [])):
                continue
            
            authors = []
            for author in entry.findall('atom:author', ns)[:5]:
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            url_elem = entry.find('atom:id', ns)
            paper_url = url_elem.text if url_elem is not None else ""
            
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=paper_url,
                source="arXiv",
                published_date=pub_date_str
            )
            paper.relevance_score = calculate_relevance(title, abstract, topics)
            papers.append(paper)
        
        logger.info(f"arXiv: {len(papers)} papers found")
        
    except Exception as e:
        logger.error(f"arXiv error: {e}")
    
    return papers

# =============================================================================
# PAPER FETCHING - OPENALEX
# =============================================================================

def fetch_openalex(config: dict) -> List[Paper]:
    """Fetch papers from OpenAlex (comprehensive academic database)."""
    if not config.get("sources", {}).get("openalex", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from OpenAlex...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    keywords = topics.get("primary", [])[:6]
    
    for keyword in keywords:
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": keyword,
                "filter": f"from_publication_date:{cutoff_str}",
                "sort": "publication_date:desc",
                "per_page": 30,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            for work in data.get("results", []):
                title = work.get("title", "")
                if not title:
                    continue
                
                # Reconstruct abstract
                abstract_inv = work.get("abstract_inverted_index", {})
                if abstract_inv:
                    word_pos = [(pos, word) for word, positions in abstract_inv.items() for pos in positions]
                    word_pos.sort()
                    abstract = " ".join(w for _, w in word_pos)
                else:
                    abstract = ""
                
                if should_exclude(title, abstract, topics.get("exclude", [])):
                    continue
                
                pub_date_str = work.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
                
                authors = [a.get("author", {}).get("display_name", "") 
                          for a in work.get("authorships", [])[:5] 
                          if a.get("author", {}).get("display_name")]
                
                # Get best URL
                paper_url = work.get("doi") or work.get("id", "")
                if paper_url and not paper_url.startswith("http"):
                    paper_url = f"https://doi.org/{paper_url}" if "/" in paper_url else paper_url
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source="OpenAlex",
                    published_date=pub_date_str
                )
                paper.relevance_score = calculate_relevance(title, abstract, topics)
                papers.append(paper)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"OpenAlex error for '{keyword}': {e}")
    
    # Deduplicate
    seen = set()
    unique = []
    for p in papers:
        key = p.title.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    logger.info(f"OpenAlex: {len(unique)} unique papers found")
    return unique

# =============================================================================
# PAPER FETCHING - SEMANTIC SCHOLAR
# =============================================================================

def fetch_semantic_scholar(config: dict) -> List[Paper]:
    """Fetch papers from Semantic Scholar."""
    if not config.get("sources", {}).get("semantic_scholar", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from Semantic Scholar...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    keywords = topics.get("primary", [])[:4]
    
    for keyword in keywords:
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": keyword,
                "limit": 25,
                "fields": "title,authors,abstract,url,publicationDate,externalIds"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            days_back = filters.get("days_lookback", 7)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for work in data.get("data", []):
                title = work.get("title", "")
                if not title:
                    continue
                
                abstract = work.get("abstract", "") or ""
                
                # Check date
                pub_date_str = work.get("publicationDate", "")
                if pub_date_str:
                    try:
                        pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                        if pub_date < cutoff_date:
                            continue
                    except:
                        pass
                else:
                    pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                if should_exclude(title, abstract, topics.get("exclude", [])):
                    continue
                
                authors = [a.get("name", "") for a in work.get("authors", [])[:5] if a.get("name")]
                
                paper_url = work.get("url", "")
                ext_ids = work.get("externalIds", {})
                if ext_ids.get("DOI"):
                    paper_url = f"https://doi.org/{ext_ids['DOI']}"
                elif ext_ids.get("ArXiv"):
                    paper_url = f"https://arxiv.org/abs/{ext_ids['ArXiv']}"
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source="Semantic Scholar",
                    published_date=pub_date_str
                )
                paper.relevance_score = calculate_relevance(title, abstract, topics)
                papers.append(paper)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Semantic Scholar error for '{keyword}': {e}")
    
    # Deduplicate
    seen = set()
    unique = []
    for p in papers:
        key = p.title.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    logger.info(f"Semantic Scholar: {len(unique)} unique papers found")
    return unique

# =============================================================================
# PAPER FETCHING - PUBMED
# =============================================================================

def fetch_pubmed(config: dict) -> List[Paper]:
    """Fetch papers from PubMed (psychology/behavioral science)."""
    if not config.get("sources", {}).get("pubmed", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from PubMed...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    
    # Build search query
    keywords = topics.get("primary", [])[:4]
    query = " OR ".join(f'"{kw}"' for kw in keywords)
    
    try:
        # Search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"({query}) AND (\"last {days_back} days\"[dp])",
            "retmax": 50,
            "retmode": "json"
        }
        
        full_url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
        
        with urllib.request.urlopen(req, timeout=30) as response:
            search_data = json.loads(response.read().decode('utf-8'))
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            logger.info("PubMed: 0 papers found")
            return []
        
        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list[:30]),
            "retmode": "xml"
        }
        
        full_url = f"{fetch_url}?{urllib.parse.urlencode(fetch_params)}"
        req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
        
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read().decode('utf-8')
        
        root = ET.fromstring(xml_data)
        
        for article in root.findall('.//PubmedArticle'):
            try:
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None and title_elem.text else ""
                if not title:
                    continue
                
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""
                
                if should_exclude(title, abstract, topics.get("exclude", [])):
                    continue
                
                # Get authors
                authors = []
                for author in article.findall('.//Author')[:5]:
                    last = author.find('LastName')
                    first = author.find('ForeName')
                    if last is not None and last.text:
                        name = last.text
                        if first is not None and first.text:
                            name = f"{first.text} {last.text}"
                        authors.append(name)
                
                # Get PMID and date
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                
                pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source="PubMed",
                    published_date=pub_date_str
                )
                paper.relevance_score = calculate_relevance(title, abstract, topics)
                papers.append(paper)
                
            except Exception as e:
                continue
        
        logger.info(f"PubMed: {len(papers)} papers found")
        
    except Exception as e:
        logger.error(f"PubMed error: {e}")
    
    return papers

# =============================================================================
# PAPER FETCHING - NBER
# =============================================================================

def fetch_nber(config: dict) -> List[Paper]:
    """Fetch NBER working papers via OpenAlex."""
    if not config.get("sources", {}).get("nber", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from NBER...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    try:
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"from_publication_date:{cutoff_str},primary_location.source.id:S4306402567",  # NBER
            "sort": "publication_date:desc",
            "per_page": 50,
            "mailto": "edimant@sas.upenn.edu"
        }
        
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        for work in data.get("results", []):
            title = work.get("title", "")
            if not title:
                continue
            
            abstract_inv = work.get("abstract_inverted_index", {})
            if abstract_inv:
                word_pos = [(pos, word) for word, positions in abstract_inv.items() for pos in positions]
                word_pos.sort()
                abstract = " ".join(w for _, w in word_pos)
            else:
                abstract = ""
            
            if should_exclude(title, abstract, topics.get("exclude", [])):
                continue
            
            # Check relevance before adding
            relevance = calculate_relevance(title, abstract, topics)
            if relevance < 0.1:
                continue
            
            pub_date_str = work.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
            
            authors = [a.get("author", {}).get("display_name", "") 
                      for a in work.get("authorships", [])[:5] 
                      if a.get("author", {}).get("display_name")]
            
            paper_url = work.get("doi") or work.get("id", "")
            if paper_url and not paper_url.startswith("http"):
                paper_url = f"https://doi.org/{paper_url}" if "/" in paper_url else paper_url
            
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=paper_url,
                source="NBER",
                published_date=pub_date_str
            )
            paper.relevance_score = relevance
            papers.append(paper)
        
        logger.info(f"NBER: {len(papers)} papers found")
        
    except Exception as e:
        logger.error(f"NBER error: {e}")
    
    return papers

# =============================================================================
# PAPER FETCHING - REPEC
# =============================================================================

def fetch_repec(config: dict) -> List[Paper]:
    """Fetch RePEc/IDEAS papers via OpenAlex."""
    if not config.get("sources", {}).get("repec", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from RePEc...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    keywords = topics.get("primary", [])[:3]
    
    for keyword in keywords:
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": keyword,
                "filter": f"from_publication_date:{cutoff_str},type:article",
                "sort": "publication_date:desc",
                "per_page": 20,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full_url, headers={'User-Agent': 'EugenResearchDigest/1.0'})
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            for work in data.get("results", []):
                # Check if it's a working paper / preprint
                work_type = work.get("type", "")
                
                title = work.get("title", "")
                if not title:
                    continue
                
                abstract_inv = work.get("abstract_inverted_index", {})
                if abstract_inv:
                    word_pos = [(pos, word) for word, positions in abstract_inv.items() for pos in positions]
                    word_pos.sort()
                    abstract = " ".join(w for _, w in word_pos)
                else:
                    abstract = ""
                
                if should_exclude(title, abstract, topics.get("exclude", [])):
                    continue
                
                pub_date_str = work.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
                
                authors = [a.get("author", {}).get("display_name", "") 
                          for a in work.get("authorships", [])[:5] 
                          if a.get("author", {}).get("display_name")]
                
                paper_url = work.get("doi") or work.get("id", "")
                if paper_url and not paper_url.startswith("http"):
                    paper_url = f"https://doi.org/{paper_url}" if "/" in paper_url else paper_url
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source="RePEc",
                    published_date=pub_date_str
                )
                paper.relevance_score = calculate_relevance(title, abstract, topics)
                papers.append(paper)
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"RePEc error for '{keyword}': {e}")
    
    # Deduplicate
    seen = set()
    unique = []
    for p in papers:
        key = p.title.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    logger.info(f"RePEc: {len(unique)} unique papers found")
    return unique

# =============================================================================
# FETCH ALL SOURCES
# =============================================================================

def fetch_all_papers(config: dict) -> List[Paper]:
    """Fetch from all enabled sources and deduplicate."""
    all_papers = []
    
    all_papers.extend(fetch_arxiv(config))
    all_papers.extend(fetch_openalex(config))
    all_papers.extend(fetch_semantic_scholar(config))
    all_papers.extend(fetch_pubmed(config))
    all_papers.extend(fetch_nber(config))
    all_papers.extend(fetch_repec(config))
    
    # Global deduplication by title similarity
    seen = set()
    unique = []
    for p in all_papers:
        key = re.sub(r'[^\w]', '', p.title.lower())[:50]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    # Filter by minimum relevance
    min_rel = config.get("filters", {}).get("min_relevance", 0.2)
    filtered = [p for p in unique if p.relevance_score >= min_rel]
    
    # Sort by relevance
    filtered.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # Filter out seen papers
    if config.get("filters", {}).get("skip_seen_papers", True):
        seen_ids = load_seen_papers()
        filtered = [p for p in filtered if p.paper_id not in seen_ids]
    
    # Limit to max papers
    max_papers = config.get("filters", {}).get("max_papers", 20)
    final = filtered[:max_papers]
    
    logger.info(f"Total: {len(all_papers)} fetched → {len(unique)} unique → {len(filtered)} new+relevant → {len(final)} final")
    return final

# =============================================================================
# AI SUMMARIZATION
# =============================================================================

def call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return ""


def generate_highlights(papers: List[Paper], config: dict) -> str:
    """Generate key highlights."""
    if not papers:
        return "No new papers found this week."
    
    if not config.get("display", {}).get("ai_summaries", True):
        return "• New research available in behavioral economics and related fields."
    
    papers_text = "\n\n".join([
        f"Title: {p.title}\nAbstract: {p.abstract[:350]}..."
        for p in papers[:8]
    ])
    
    prompt = f"""You're creating a research digest for Professor Eugen Dimant, a behavioral economist at UPenn/Stanford studying dishonesty, social norms, and experimental economics.

Based on these papers, write 2-3 key highlights. Each should be 2 sentences, starting with **Bold Theme**:

Papers:
{papers_text}

Write only bullet points with •, no preamble."""

    result = call_gemini(prompt)
    return result if result else "• **New Research**: Papers this week cover topics relevant to your behavioral economics research."


def generate_opportunities(papers: List[Paper], config: dict) -> str:
    """Generate research opportunities."""
    if not config.get("display", {}).get("show_opportunities", True):
        return ""
    
    if not papers:
        return ""
    
    papers_text = "\n".join([f"- {p.title}" for p in papers[:6]])
    
    prompt = f"""Based on these recent papers, suggest 1-2 potential research opportunities for a behavioral economist studying dishonesty and social norms. Be specific and actionable (3 sentences max):

{papers_text}"""

    return call_gemini(prompt)


def summarize_paper(paper: Paper) -> str:
    """Generate paper summary."""
    prompt = f"""Summarize for a behavioral economist in 2 sentences. Focus on: finding, method, relevance to dishonesty/norms.

Title: {paper.title}
Abstract: {paper.abstract[:500]}

Write only the summary."""

    result = call_gemini(prompt)
    return result if result else paper.abstract[:200] + "..."

# =============================================================================
# CATEGORIZATION
# =============================================================================

def categorize_papers(papers: List[Paper]) -> Dict[str, List[Paper]]:
    """Categorize papers."""
    categories = {
        "Dishonesty & Ethics": [],
        "Social Norms & Cooperation": [],
        "Experimental Methods": [],
        "Policy & Interventions": [],
        "Other Relevant": []
    }
    
    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        
        if any(kw in text for kw in ["dishonest", "cheat", "corrupt", "unethical", "fraud", "lying", "deception", "honesty"]):
            categories["Dishonesty & Ethics"].append(paper)
        elif any(kw in text for kw in ["norm", "cooperat", "trust", "prosocial", "public good", "collective"]):
            categories["Social Norms & Cooperation"].append(paper)
        elif any(kw in text for kw in ["experiment", "rct", "randomized", "lab study", "field study", "treatment effect"]):
            categories["Experimental Methods"].append(paper)
        elif any(kw in text for kw in ["policy", "intervention", "nudge", "regulation", "incentive"]):
            categories["Policy & Interventions"].append(paper)
        else:
            categories["Other Relevant"].append(paper)
    
    return {k: v for k, v in categories.items() if v}

# =============================================================================
# EMAIL GENERATION
# =============================================================================

def format_paper_html(paper: Paper, config: dict) -> str:
    """Format paper as HTML."""
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."
    
    badge = ""
    if config.get("display", {}).get("show_relevance_badges", True):
        if paper.relevance_score >= 0.5:
            badge = '<span style="background-color: #059669; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 8px;">High Match</span>'
        elif paper.relevance_score >= 0.3:
            badge = '<span style="background-color: #d97706; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 8px;">Good Match</span>'
    
    summary = paper.summary if paper.summary else paper.abstract[:280] + "..."
    
    return f"""
    <div style="margin-bottom: 20px; padding: 16px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
        <h3 style="margin: 0 0 8px 0; font-size: 15px; line-height: 1.4;">
            <a href="{paper.url}" style="color: #1e40af; text-decoration: none;">{paper.title}</a>
            {badge}
        </h3>
        <p style="margin: 0 0 8px 0; color: #64748b; font-size: 12px;">
            {authors_str} • <span style="background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{paper.source}</span> • {paper.published_date}
        </p>
        <p style="margin: 0; color: #475569; font-size: 13px; line-height: 1.5;">{summary}</p>
    </div>
    """


def generate_email_html(papers: List[Paper], highlights: str, opportunities: str, 
                        categories: Dict[str, List[Paper]], config: dict) -> str:
    """Generate HTML email."""
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
        highlight_items.append(f'<li style="margin-bottom: 10px; line-height: 1.5;">{line}</li>')
    highlights_html = f'<ul style="margin: 0; padding-left: 20px; color: #374151;">{" ".join(highlight_items)}</ul>'
    
    # Opportunities
    opportunities_html = ""
    if opportunities:
        opportunities_html = f"""
        <div style="margin: 24px 0; padding: 16px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h3 style="margin: 0 0 10px 0; font-size: 15px; color: #92400e;">💡 Research Opportunities</h3>
            <p style="margin: 0; color: #78350f; font-size: 13px; line-height: 1.5;">{opportunities}</p>
        </div>
        """
    
    # Papers
    if config.get("display", {}).get("group_by_category", True) and categories:
        papers_html = ""
        for cat_name, cat_papers in categories.items():
            cat_papers_html = "".join([format_paper_html(p, config) for p in cat_papers])
            papers_html += f"""
            <div style="margin-bottom: 28px;">
                <h2 style="margin: 0 0 16px 0; font-size: 17px; color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">
                    {cat_name} <span style="color: #64748b; font-weight: normal;">({len(cat_papers)})</span>
                </h2>
                {cat_papers_html}
            </div>
            """
    else:
        papers_html = "".join([format_paper_html(p, config) for p in papers])
    
    sources = sorted(set(p.source for p in papers))
    
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f1f5f9; margin: 0; padding: 20px;">
    <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%); padding: 36px 32px; text-align: center;">
            <h1 style="margin: 0; color: white; font-size: 26px; font-weight: 600; letter-spacing: -0.5px;">
                Eugen's Weekly<br>Research Digest
            </h1>
            <p style="margin: 12px 0 0 0; color: #93c5fd; font-size: 14px;">{date_str}</p>
        </div>
        
        <!-- Summary -->
        <div style="margin: 24px; padding: 20px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <h2 style="margin: 0 0 12px 0; font-size: 17px; color: #1e40af;">📊 This Week's Highlights</h2>
            <p style="margin: 0 0 14px 0; color: #1e40af; font-size: 14px;">
                <strong>{len(papers)} new papers</strong> from {', '.join(sources)}
            </p>
            {highlights_html}
        </div>
        
        <!-- Opportunities -->
        <div style="margin: 0 24px;">
            {opportunities_html}
        </div>
        
        <!-- Papers -->
        <div style="padding: 0 24px 24px 24px;">
            {papers_html}
        </div>
        
        <!-- Footer -->
        <div style="background: #f8fafc; padding: 20px 24px; text-align: center; border-top: 1px solid #e2e8f0;">
            <p style="margin: 0 0 6px 0; color: #64748b; font-size: 12px;">
                Automatically generated for Eugen Dimant • Only new papers since last digest
            </p>
            <p style="margin: 0; color: #94a3b8; font-size: 11px;">
                <a href="YOUR_STREAMLIT_URL" style="color: #3b82f6;">Manage Settings</a> • 
                Sources: arXiv, OpenAlex, Semantic Scholar, PubMed, NBER, RePEc
            </p>
        </div>
    </div>
</body>
</html>"""

# =============================================================================
# EMAIL SENDING
# =============================================================================

def send_email(html: str, config: dict) -> bool:
    """Send email via SMTP."""
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    recipient = config.get("email", {}).get("recipient", "")
    
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
        
        logger.info(f"Email sent to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# =============================================================================
# TEST DATA
# =============================================================================

def create_test_papers() -> List[Paper]:
    """Create test papers for testing."""
    return [
        Paper(
            title="Social Norms and Tax Compliance: Evidence from a Large-Scale Field Experiment",
            authors=["Maria Garcia", "John Smith", "Wei Chen"],
            abstract="We conduct a field experiment with 50,000 taxpayers testing how social norm messages affect compliance. Descriptive norms ('most people pay taxes honestly') increased compliance by 12%, while injunctive norms ('tax evasion is wrong') had smaller effects. Results suggest descriptive framing is more effective for behavioral interventions.",
            url="https://doi.org/10.1234/example1",
            source="NBER",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            relevance_score=0.85,
            summary="Field experiment (N=50,000) shows descriptive social norms increase tax compliance by 12%, outperforming injunctive norm messages. Implications for designing behavioral policy interventions."
        ),
        Paper(
            title="The Psychology of Lying: When Do People Justify Dishonesty?",
            authors=["Sarah Johnson", "Michael Brown"],
            abstract="Five experiments examine moral justification of lies. We find that perceived harm to others, not self-benefit, predicts when people justify dishonesty. Altruistic lies are judged more acceptable than selfish lies, even when outcomes are identical.",
            url="https://doi.org/10.1234/example2",
            source="OpenAlex",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            relevance_score=0.78,
            summary="Series of experiments reveals people justify lies based on perceived harm to others rather than personal gain. Altruistic lies are more acceptable than selfish ones regardless of outcomes."
        ),
        Paper(
            title="Peer Effects in Cooperative Behavior: A Multi-Country Experiment",
            authors=["Emma Wilson", "David Lee", "Anna Mueller"],
            abstract="Using public goods games across 25 countries, we examine how peer behavior affects cooperation. Observing cooperative peers increases contributions by 23%, with effects stronger in collectivist cultures. Results highlight cultural moderation of social influence.",
            url="https://doi.org/10.1234/example3",
            source="arXiv",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            relevance_score=0.72,
            summary="Cross-cultural public goods experiment finds peer cooperation increases contributions 23%, with stronger effects in collectivist societies. Cultural context moderates social influence on prosocial behavior."
        )
    ]

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the digest pipeline."""
    test_mode = "--test" in sys.argv or os.environ.get("TEST_EMAIL") == "true"
    preview_mode = "--preview" in sys.argv
    
    logger.info(f"Starting Eugen's Research Digest (test={test_mode}, preview={preview_mode})")
    
    config = load_config()
    
    if test_mode:
        papers = create_test_papers()
        highlights = "• **Behavioral Interventions**: New field experiments show social norm messaging can significantly impact real-world behavior like tax compliance.\n• **Psychology of Dishonesty**: Research reveals people justify lies based on harm to others rather than self-interest."
        opportunities = "The tax compliance methodology could be adapted for studying honesty interventions in organizational contexts. Consider cross-cultural extensions of the peer effects findings."
    else:
        papers = fetch_all_papers(config)
        
        if not papers:
            highlights = "No new papers matching your interests were found this week."
            opportunities = ""
        else:
            # Generate AI content
            if config.get("display", {}).get("ai_summaries", True):
                highlights = generate_highlights(papers, config)
                opportunities = generate_opportunities(papers, config)
                for paper in papers[:12]:
                    paper.summary = summarize_paper(paper)
            else:
                highlights = f"• {len(papers)} new papers found across your research areas."
                opportunities = ""
    
    # Categorize
    categories = categorize_papers(papers) if config.get("display", {}).get("group_by_category", True) else {}
    
    # Generate email
    html = generate_email_html(papers, highlights, opportunities, categories, config)
    
    # Save preview
    with open("latest_digest.html", "w") as f:
        f.write(html)
    logger.info("Saved preview to latest_digest.html")
    
    # Save papers JSON
    with open("latest_papers.json", "w") as f:
        json.dump([p.to_dict() for p in papers], f, indent=2)
    
    if preview_mode:
        logger.info("Preview mode - email not sent")
        return True
    
    # Send email
    success = send_email(html, config)
    
    # Update seen papers
    if success and not test_mode:
        seen = load_seen_papers()
        for p in papers:
            seen.add(p.paper_id)
        save_seen_papers(seen)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
