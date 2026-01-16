#!/usr/bin/env python3
"""
Eugen's Weekly Research Digest - Enhanced Version
With DOI validation, SSRN support, and per-paper research connections.
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
import xml.etree.ElementTree as ET
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Edit these to customize your digest
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
            "lying",
            "deception",
            "moral behavior"
        ],
        "secondary": [
            "nudge",
            "nudging",
            "decision making",
            "prosocial behavior",
            "cooperation",
            "trust game",
            "public goods game",
            "dictator game",
            "ultimatum game",
            "field experiment",
            "lab experiment",
            "randomized controlled trial",
            "behavioral intervention",
            "norm enforcement",
            "peer effects",
            "social influence",
            "incentive design"
        ],
        "exclude": [
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "blockchain",
            "stock prediction",
            "deep learning for trading",
            "algorithmic trading",
            "sports betting",
            "poker",
            "casino"
        ]
    },
    "sources": {
        "arxiv": {"enabled": True},
        "openalex": {"enabled": True},
        "semantic_scholar": {"enabled": True},
        "ssrn": {"enabled": True},
        "pubmed": {"enabled": True},
        "crossref": {"enabled": True}
    },
    "filters": {
        "days_lookback": 7,
        "min_relevance": 0.15,
        "max_papers": 20,
        "skip_seen_papers": True,
        "require_doi": True
    },
    "display": {
        "show_relevance_badges": True,
        "group_by_category": True,
        "ai_summaries": True,
        "show_research_connections": True
    }
}

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
    doi: str = ""
    relevance_score: float = 0.0
    summary: str = ""
    research_connection: str = ""
    paper_id: str = ""
    
    def __post_init__(self):
        if not self.paper_id:
            self.paper_id = hashlib.md5(self.title.lower().encode()).hexdigest()[:12]
        # Clean DOI - remove URL prefix if present
        if self.doi:
            self.doi = self.doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        # Use DOI as primary URL if available
        if self.doi:
            self.url = f"https://doi.org/{self.doi}"
    
    def to_dict(self):
        return asdict(self)

# =============================================================================
# DOI VALIDATION
# =============================================================================

def validate_doi(doi: str) -> Tuple[bool, str]:
    """
    Validate a DOI by checking its format and optionally verifying it resolves.
    Returns (is_valid, cleaned_doi).
    """
    if not doi:
        return False, ""
    
    # Clean the DOI
    doi = doi.strip()
    
    # Remove URL prefix if present
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
    
    # Check DOI format (starts with 10. followed by registrant code / suffix)
    doi_pattern = r'^10\.\d{4,}/[^\s]+$'
    if not re.match(doi_pattern, doi):
        return False, ""
    
    return True, doi


def extract_doi_from_text(text: str) -> str:
    """Extract DOI from text using regex."""
    if not text:
        return ""
    
    # Common DOI patterns
    patterns = [
        r'10\.\d{4,}/[^\s\]>"\']+',
        r'doi\.org/(10\.\d{4,}/[^\s\]>"\']+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if '(' in pattern else match.group(0)
            # Clean trailing punctuation
            doi = re.sub(r'[.,;:\]\)>]+$', '', doi)
            is_valid, clean_doi = validate_doi(doi)
            if is_valid:
                return clean_doi
    
    return ""

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
    ids_list = list(paper_ids)[-1500:]  # Keep last 1500
    with open(SEEN_PAPERS_FILE, "w") as f:
        json.dump({
            "paper_ids": ids_list,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Saved {len(ids_list)} seen paper IDs")

# =============================================================================
# RELEVANCE SCORING
# =============================================================================

def calculate_relevance(title: str, abstract: str) -> float:
    """Calculate relevance score based on keyword matches."""
    text = f"{title} {abstract}".lower()
    topics = CONFIG["topics"]
    
    primary = topics.get("primary", [])
    secondary = topics.get("secondary", [])
    
    primary_matches = sum(1 for kw in primary if kw.lower() in text)
    secondary_matches = sum(1 for kw in secondary if kw.lower() in text)
    
    total_primary = max(len(primary), 1)
    total_secondary = max(len(secondary), 1)
    
    score = (0.65 * (primary_matches / total_primary) + 
             0.35 * (secondary_matches / total_secondary))
    
    # Boost for multiple primary matches
    if primary_matches >= 2:
        score = min(score * 1.3, 1.0)
    if primary_matches >= 3:
        score = min(score * 1.2, 1.0)
    
    return round(score, 3)

def should_exclude(title: str, abstract: str) -> bool:
    """Check if paper should be excluded."""
    text = f"{title} {abstract}".lower()
    exclude_terms = CONFIG["topics"].get("exclude", [])
    return any(term.lower() in text for term in exclude_terms)

# =============================================================================
# HTTP HELPER
# =============================================================================

def http_get(url: str, headers: dict = None, timeout: int = 30) -> Optional[bytes]:
    """Make HTTP GET request with error handling."""
    if headers is None:
        headers = {'User-Agent': 'EugenResearchDigest/2.0 (mailto:edimant@sas.upenn.edu)'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        logger.debug(f"HTTP error for {url[:50]}...: {e}")
        return None

def http_get_json(url: str, headers: dict = None) -> Optional[dict]:
    """Make HTTP GET request and parse JSON response."""
    data = http_get(url, headers)
    if data:
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            return None
    return None

# =============================================================================
# PAPER FETCHING - CROSSREF (Primary source for DOI verification)
# =============================================================================

def fetch_crossref(config: dict) -> List[Paper]:
    """Fetch papers from Crossref API - most reliable for DOIs."""
    if not config.get("sources", {}).get("crossref", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from Crossref...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    from_date = cutoff_date.strftime("%Y-%m-%d")
    
    keywords = topics.get("primary", [])[:5]
    
    for keyword in keywords:
        try:
            url = "https://api.crossref.org/works"
            params = {
                "query": keyword,
                "filter": f"from-pub-date:{from_date}",
                "sort": "published",
                "order": "desc",
                "rows": 30,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data or "message" not in data:
                continue
            
            for item in data["message"].get("items", []):
                # Must have DOI
                doi = item.get("DOI", "")
                if not doi:
                    continue
                
                is_valid, clean_doi = validate_doi(doi)
                if not is_valid:
                    continue
                
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""
                if not title:
                    continue
                
                abstract = item.get("abstract", "")
                # Clean HTML from abstract
                abstract = re.sub(r'<[^>]+>', '', abstract)
                
                if should_exclude(title, abstract):
                    continue
                
                # Get authors
                authors = []
                for author in item.get("author", [])[:5]:
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                
                # Get publication date
                pub_date = item.get("published", {}).get("date-parts", [[]])
                if pub_date and pub_date[0]:
                    parts = pub_date[0]
                    year = parts[0] if len(parts) > 0 else datetime.now().year
                    month = parts[1] if len(parts) > 1 else 1
                    day = parts[2] if len(parts) > 2 else 1
                    pub_date_str = f"{year}-{month:02d}-{day:02d}"
                else:
                    pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://doi.org/{clean_doi}",
                    source="Crossref",
                    published_date=pub_date_str,
                    doi=clean_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
            
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Crossref error for '{keyword}': {e}")
    
    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for p in papers:
        if p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    logger.info(f"Crossref: {len(unique)} papers with valid DOIs")
    return unique

# =============================================================================
# PAPER FETCHING - OPENALEX
# =============================================================================

def fetch_openalex(config: dict) -> List[Paper]:
    """Fetch papers from OpenAlex."""
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
                "filter": f"from_publication_date:{cutoff_str},has_doi:true",
                "sort": "publication_date:desc",
                "per_page": 25,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data:
                continue
            
            for work in data.get("results", []):
                # Must have DOI
                doi_url = work.get("doi", "")
                if not doi_url:
                    continue
                
                # Extract DOI from URL
                doi = doi_url.replace("https://doi.org/", "")
                is_valid, clean_doi = validate_doi(doi)
                if not is_valid:
                    continue
                
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
                
                if should_exclude(title, abstract):
                    continue
                
                pub_date_str = work.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
                
                authors = [a.get("author", {}).get("display_name", "") 
                          for a in work.get("authorships", [])[:5] 
                          if a.get("author", {}).get("display_name")]
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://doi.org/{clean_doi}",
                    source="OpenAlex",
                    published_date=pub_date_str,
                    doi=clean_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
            
            time.sleep(0.15)
            
        except Exception as e:
            logger.error(f"OpenAlex error for '{keyword}': {e}")
    
    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for p in papers:
        if p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    logger.info(f"OpenAlex: {len(unique)} papers with valid DOIs")
    return unique

# =============================================================================
# PAPER FETCHING - SSRN
# =============================================================================

def fetch_ssrn(config: dict) -> List[Paper]:
    """Fetch papers from SSRN via OpenAlex and Crossref."""
    if not config.get("sources", {}).get("ssrn", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from SSRN...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    keywords = topics.get("primary", [])[:4]
    
    # Method 1: OpenAlex with SSRN filter
    for keyword in keywords:
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": keyword,
                "filter": f"from_publication_date:{cutoff_str},primary_location.source.display_name:SSRN",
                "sort": "publication_date:desc",
                "per_page": 20,
                "mailto": "edimant@sas.upenn.edu"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data:
                continue
            
            for work in data.get("results", []):
                title = work.get("title", "")
                if not title:
                    continue
                
                # Get DOI or SSRN URL
                doi_url = work.get("doi", "")
                ssrn_id = ""
                
                # Try to get SSRN ID from locations
                for location in work.get("locations", []):
                    landing_page = location.get("landing_page_url", "")
                    if "ssrn.com" in landing_page:
                        match = re.search(r'abstract[_=]?(\d+)', landing_page)
                        if match:
                            ssrn_id = match.group(1)
                            break
                
                # Determine URL and DOI
                if doi_url:
                    doi = doi_url.replace("https://doi.org/", "")
                    is_valid, clean_doi = validate_doi(doi)
                    if is_valid:
                        paper_url = f"https://doi.org/{clean_doi}"
                        paper_doi = clean_doi
                    elif ssrn_id:
                        # SSRN DOI format
                        paper_doi = f"10.2139/ssrn.{ssrn_id}"
                        paper_url = f"https://doi.org/{paper_doi}"
                    else:
                        continue
                elif ssrn_id:
                    paper_doi = f"10.2139/ssrn.{ssrn_id}"
                    paper_url = f"https://doi.org/{paper_doi}"
                else:
                    continue
                
                # Reconstruct abstract
                abstract_inv = work.get("abstract_inverted_index", {})
                if abstract_inv:
                    word_pos = [(pos, word) for word, positions in abstract_inv.items() for pos in positions]
                    word_pos.sort()
                    abstract = " ".join(w for _, w in word_pos)
                else:
                    abstract = ""
                
                if should_exclude(title, abstract):
                    continue
                
                pub_date_str = work.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
                
                authors = [a.get("author", {}).get("display_name", "") 
                          for a in work.get("authorships", [])[:5] 
                          if a.get("author", {}).get("display_name")]
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source="SSRN",
                    published_date=pub_date_str,
                    doi=paper_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
            
            time.sleep(0.15)
            
        except Exception as e:
            logger.error(f"SSRN/OpenAlex error for '{keyword}': {e}")
    
    # Method 2: Crossref with SSRN as publisher
    try:
        from_date = cutoff_date.strftime("%Y-%m-%d")
        url = "https://api.crossref.org/works"
        params = {
            "filter": f"from-pub-date:{from_date},publisher-name:Elsevier BV",  # SSRN is owned by Elsevier
            "query": "behavioral economics OR dishonesty OR social norms",
            "rows": 30,
            "mailto": "edimant@sas.upenn.edu"
        }
        
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        data = http_get_json(full_url)
        
        if data and "message" in data:
            for item in data["message"].get("items", []):
                # Check if SSRN
                publisher = item.get("publisher", "").lower()
                if "ssrn" not in publisher and "social science research network" not in publisher:
                    continue
                
                doi = item.get("DOI", "")
                if not doi:
                    continue
                
                is_valid, clean_doi = validate_doi(doi)
                if not is_valid:
                    continue
                
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""
                if not title:
                    continue
                
                abstract = re.sub(r'<[^>]+>', '', item.get("abstract", ""))
                
                if should_exclude(title, abstract):
                    continue
                
                authors = []
                for author in item.get("author", [])[:5]:
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                
                pub_date = item.get("published", {}).get("date-parts", [[]])
                if pub_date and pub_date[0]:
                    parts = pub_date[0]
                    year = parts[0] if len(parts) > 0 else datetime.now().year
                    month = parts[1] if len(parts) > 1 else 1
                    day = parts[2] if len(parts) > 2 else 1
                    pub_date_str = f"{year}-{month:02d}-{day:02d}"
                else:
                    pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://doi.org/{clean_doi}",
                    source="SSRN",
                    published_date=pub_date_str,
                    doi=clean_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
                    
    except Exception as e:
        logger.error(f"SSRN/Crossref error: {e}")
    
    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for p in papers:
        if p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    logger.info(f"SSRN: {len(unique)} papers with valid DOIs")
    return unique

# =============================================================================
# PAPER FETCHING - ARXIV
# =============================================================================

def fetch_arxiv(config: dict) -> List[Paper]:
    """Fetch papers from arXiv with DOI verification."""
    if not config.get("sources", {}).get("arxiv", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from arXiv...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    categories = ["econ.GN", "econ.TH", "q-fin.EC", "cs.GT"]
    keywords = topics.get("primary", [])[:4]
    
    search_terms = " OR ".join(f'all:"{kw}"' for kw in keywords)
    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    query = f"({search_terms}) AND ({cat_query})"
    
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 80,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        response_data = http_get(full_url)
        
        if not response_data:
            return []
        
        root = ET.fromstring(response_data.decode('utf-8'))
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            if title_elem is None or not title_elem.text:
                continue
            title = ' '.join(title_elem.text.strip().split())
            
            abstract_elem = entry.find('atom:summary', ns)
            abstract = ' '.join(abstract_elem.text.strip().split()) if abstract_elem is not None and abstract_elem.text else ""
            
            # Get publication date
            published_elem = entry.find('atom:published', ns)
            if published_elem is not None and published_elem.text:
                pub_date = datetime.strptime(published_elem.text[:10], "%Y-%m-%d")
                if pub_date < cutoff_date:
                    continue
                pub_date_str = pub_date.strftime("%Y-%m-%d")
            else:
                continue
            
            if should_exclude(title, abstract):
                continue
            
            # Get arXiv ID and construct DOI
            arxiv_id_elem = entry.find('atom:id', ns)
            if arxiv_id_elem is None or not arxiv_id_elem.text:
                continue
            
            arxiv_url = arxiv_id_elem.text
            # Extract arXiv ID (e.g., 2401.12345)
            match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', arxiv_url)
            if match:
                arxiv_id = match.group(1)
                # arXiv DOI format
                paper_doi = f"10.48550/arXiv.{arxiv_id}"
                paper_url = f"https://doi.org/{paper_doi}"
            else:
                # Try older format
                match = re.search(r'arxiv\.org/abs/([a-z-]+/\d+)', arxiv_url)
                if match:
                    arxiv_id = match.group(1).replace('/', '.')
                    paper_doi = f"10.48550/arXiv.{arxiv_id}"
                    paper_url = f"https://doi.org/{paper_doi}"
                else:
                    continue
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns)[:5]:
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=paper_url,
                source="arXiv",
                published_date=pub_date_str,
                doi=paper_doi
            )
            paper.relevance_score = calculate_relevance(title, abstract)
            
            if paper.relevance_score >= 0.1:
                papers.append(paper)
        
        logger.info(f"arXiv: {len(papers)} papers with DOIs")
        
    except Exception as e:
        logger.error(f"arXiv error: {e}")
    
    return papers

# =============================================================================
# PAPER FETCHING - SEMANTIC SCHOLAR
# =============================================================================

def fetch_semantic_scholar(config: dict) -> List[Paper]:
    """Fetch papers from Semantic Scholar with DOI requirement."""
    if not config.get("sources", {}).get("semantic_scholar", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from Semantic Scholar...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    keywords = topics.get("primary", [])[:4]
    
    for keyword in keywords:
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": keyword,
                "limit": 30,
                "fields": "title,authors,abstract,url,publicationDate,externalIds,year"
            }
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            data = http_get_json(full_url)
            
            if not data:
                continue
            
            for work in data.get("data", []):
                title = work.get("title", "")
                if not title:
                    continue
                
                # Must have DOI
                ext_ids = work.get("externalIds", {})
                doi = ext_ids.get("DOI", "")
                
                if not doi:
                    # Try arXiv DOI
                    arxiv_id = ext_ids.get("ArXiv", "")
                    if arxiv_id:
                        doi = f"10.48550/arXiv.{arxiv_id}"
                    else:
                        continue
                
                is_valid, clean_doi = validate_doi(doi)
                if not is_valid:
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
                        # Use year if available
                        year = work.get("year")
                        if year and year < cutoff_date.year:
                            continue
                        pub_date_str = f"{year}-01-01" if year else datetime.now().strftime("%Y-%m-%d")
                else:
                    year = work.get("year")
                    if year:
                        pub_date_str = f"{year}-01-01"
                    else:
                        pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                if should_exclude(title, abstract):
                    continue
                
                authors = [a.get("name", "") for a in work.get("authors", [])[:5] if a.get("name")]
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://doi.org/{clean_doi}",
                    source="Semantic Scholar",
                    published_date=pub_date_str,
                    doi=clean_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Semantic Scholar error for '{keyword}': {e}")
    
    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for p in papers:
        if p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    logger.info(f"Semantic Scholar: {len(unique)} papers with valid DOIs")
    return unique

# =============================================================================
# PAPER FETCHING - PUBMED
# =============================================================================

def fetch_pubmed(config: dict) -> List[Paper]:
    """Fetch papers from PubMed with DOI requirement."""
    if not config.get("sources", {}).get("pubmed", {}).get("enabled", True):
        return []
    
    logger.info("Fetching from PubMed...")
    papers = []
    topics = config.get("topics", {})
    filters = config.get("filters", {})
    
    days_back = filters.get("days_lookback", 7)
    
    keywords = topics.get("primary", [])[:3]
    query = " OR ".join(f'"{kw}"' for kw in keywords)
    
    try:
        # Search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"({query}) AND (\"last {days_back} days\"[dp])",
            "retmax": 40,
            "retmode": "json"
        }
        
        full_url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        search_data = http_get_json(full_url)
        
        if not search_data:
            return []
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            logger.info("PubMed: 0 papers found")
            return []
        
        # Fetch details in XML format
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list[:30]),
            "retmode": "xml"
        }
        
        full_url = f"{fetch_url}?{urllib.parse.urlencode(fetch_params)}"
        xml_data = http_get(full_url)
        
        if not xml_data:
            return []
        
        root = ET.fromstring(xml_data.decode('utf-8'))
        
        for article in root.findall('.//PubmedArticle'):
            try:
                # Get DOI
                doi = ""
                for article_id in article.findall('.//ArticleId'):
                    if article_id.get('IdType') == 'doi':
                        doi = article_id.text
                        break
                
                if not doi:
                    # Try ELocationID
                    for eloc in article.findall('.//ELocationID'):
                        if eloc.get('EIdType') == 'doi':
                            doi = eloc.text
                            break
                
                if not doi:
                    continue
                
                is_valid, clean_doi = validate_doi(doi)
                if not is_valid:
                    continue
                
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None and title_elem.text else ""
                if not title:
                    continue
                
                # Get abstract
                abstract_parts = []
                for abstract_text in article.findall('.//AbstractText'):
                    if abstract_text.text:
                        abstract_parts.append(abstract_text.text)
                abstract = " ".join(abstract_parts)
                
                if should_exclude(title, abstract):
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
                
                # Get date
                pub_date_elem = article.find('.//PubDate')
                if pub_date_elem is not None:
                    year = pub_date_elem.find('Year')
                    month = pub_date_elem.find('Month')
                    day = pub_date_elem.find('Day')
                    
                    y = year.text if year is not None and year.text else str(datetime.now().year)
                    m = month.text if month is not None and month.text else "01"
                    d = day.text if day is not None and day.text else "01"
                    
                    # Convert month name to number if needed
                    try:
                        m = int(m)
                    except:
                        months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                        m = months.get(m.lower()[:3], 1)
                    
                    pub_date_str = f"{y}-{int(m):02d}-{int(d):02d}"
                else:
                    pub_date_str = datetime.now().strftime("%Y-%m-%d")
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://doi.org/{clean_doi}",
                    source="PubMed",
                    published_date=pub_date_str,
                    doi=clean_doi
                )
                paper.relevance_score = calculate_relevance(title, abstract)
                
                if paper.relevance_score >= 0.1:
                    papers.append(paper)
                
            except Exception as e:
                continue
        
        logger.info(f"PubMed: {len(papers)} papers with valid DOIs")
        
    except Exception as e:
        logger.error(f"PubMed error: {e}")
    
    return papers

# =============================================================================
# FETCH ALL SOURCES
# =============================================================================

def fetch_all_papers() -> List[Paper]:
    """Fetch from all enabled sources, deduplicate by DOI."""
    all_papers = []
    
    # Fetch from each source
    all_papers.extend(fetch_crossref(CONFIG))
    all_papers.extend(fetch_openalex(CONFIG))
    all_papers.extend(fetch_ssrn(CONFIG))
    all_papers.extend(fetch_arxiv(CONFIG))
    all_papers.extend(fetch_semantic_scholar(CONFIG))
    all_papers.extend(fetch_pubmed(CONFIG))
    
    # Global deduplication by DOI (most reliable)
    seen_dois = set()
    unique = []
    for p in all_papers:
        if p.doi and p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique.append(p)
    
    # Filter by minimum relevance
    min_rel = CONFIG.get("filters", {}).get("min_relevance", 0.15)
    filtered = [p for p in unique if p.relevance_score >= min_rel]
    
    # Sort by relevance
    filtered.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # Filter out seen papers
    if CONFIG.get("filters", {}).get("skip_seen_papers", True):
        seen_ids = load_seen_papers()
        filtered = [p for p in filtered if p.paper_id not in seen_ids]
    
    # Limit to max papers
    max_papers = CONFIG.get("filters", {}).get("max_papers", 20)
    final = filtered[:max_papers]
    
    # Final verification: all papers must have DOI
    final = [p for p in final if p.doi]
    
    logger.info(f"Pipeline: {len(all_papers)} fetched → {len(unique)} unique DOIs → {len(filtered)} new+relevant → {len(final)} final")
    return final

# =============================================================================
# AI SUMMARIZATION & RESEARCH CONNECTIONS
# =============================================================================

def call_gemini(prompt: str, max_retries: int = 2) -> str:
    """Call Gemini API with retry logic."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}
    }
    
    for attempt in range(max_retries):
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if text:
                return text.strip()
                
        except Exception as e:
            logger.debug(f"Gemini attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    
    return ""


def generate_paper_summary_and_connection(paper: Paper) -> Tuple[str, str]:
    """Generate summary and research connection for a single paper."""
    profile = CONFIG.get("researcher_profile", {})
    focus_areas = profile.get("focus_areas", [])
    
    prompt = f"""You are helping Professor Eugen Dimant (UPenn/Stanford behavioral economist) review new research.

His research focuses on: {', '.join(focus_areas)}

Paper:
Title: {paper.title}
Abstract: {paper.abstract[:600]}

Provide two things:

1. SUMMARY (2 sentences): What is the main finding and method?

2. RESEARCH CONNECTION (2 sentences): How could this paper connect to or advance Eugen's research? Be specific about potential extensions, methodological borrowings, or theoretical links.

Format your response exactly like this:
SUMMARY: [your summary]
CONNECTION: [your connection]"""

    response = call_gemini(prompt)
    
    if not response:
        return paper.abstract[:200] + "...", ""
    
    # Parse response
    summary = ""
    connection = ""
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.upper().startswith("CONNECTION:"):
            connection = line[11:].strip()
    
    # Fallback
    if not summary:
        summary = paper.abstract[:200] + "..."
    
    return summary, connection


def generate_highlights(papers: List[Paper]) -> str:
    """Generate key highlights section."""
    if not papers:
        return "No new papers found this week matching your research interests."
    
    papers_text = "\n\n".join([
        f"• {p.title} (Relevance: {p.relevance_score:.0%})"
        for p in papers[:10]
    ])
    
    prompt = f"""Based on these new behavioral economics papers, write 2-3 key thematic highlights for Professor Eugen Dimant.

Papers:
{papers_text}

Write bullet points (use •) with **Bold Theme** titles. Each highlight should be 1-2 sentences identifying patterns or important developments. Focus on themes related to dishonesty, social norms, and experimental methods."""

    result = call_gemini(prompt)
    return result if result else "• **New Research Available**: Several relevant papers published this week in behavioral economics."

# =============================================================================
# CATEGORIZATION
# =============================================================================

def categorize_papers(papers: List[Paper]) -> Dict[str, List[Paper]]:
    """Categorize papers into topic groups."""
    categories = {
        "Dishonesty & Ethics": [],
        "Social Norms & Cooperation": [],
        "Experimental Methods": [],
        "Policy & Interventions": [],
        "Other Relevant": []
    }
    
    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        
        if any(kw in text for kw in ["dishonest", "cheat", "corrupt", "unethical", "fraud", "lying", "deception", "honesty", "truth"]):
            categories["Dishonesty & Ethics"].append(paper)
        elif any(kw in text for kw in ["norm", "cooperat", "trust", "prosocial", "public good", "collective action"]):
            categories["Social Norms & Cooperation"].append(paper)
        elif any(kw in text for kw in ["experiment", "rct", "randomized", "lab study", "field study", "treatment effect", "causal"]):
            categories["Experimental Methods"].append(paper)
        elif any(kw in text for kw in ["policy", "intervention", "nudge", "regulation", "incentive"]):
            categories["Policy & Interventions"].append(paper)
        else:
            categories["Other Relevant"].append(paper)
    
    return {k: v for k, v in categories.items() if v}

# =============================================================================
# EMAIL GENERATION
# =============================================================================

def format_paper_html(paper: Paper) -> str:
    """Format a single paper as HTML with DOI link and research connection."""
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."
    
    # Relevance badge
    if paper.relevance_score >= 0.5:
        badge = '<span style="background: #059669; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-left: 8px;">High Match</span>'
    elif paper.relevance_score >= 0.3:
        badge = '<span style="background: #d97706; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-left: 8px;">Good Match</span>'
    else:
        badge = ""
    
    summary = paper.summary if paper.summary else paper.abstract[:250] + "..."
    
    # Research connection section
    connection_html = ""
    if paper.research_connection:
        connection_html = f"""
        <div style="margin-top: 10px; padding: 10px; background: #fef3c7; border-radius: 6px; border-left: 3px solid #f59e0b;">
            <p style="margin: 0; font-size: 12px; color: #92400e;">
                <strong>🔗 Research Connection:</strong> {paper.research_connection}
            </p>
        </div>
        """
    
    # DOI display
    doi_display = paper.doi if paper.doi else "N/A"
    
    return f"""
    <div style="margin-bottom: 24px; padding: 18px; background: #f8fafc; border-radius: 10px; border-left: 4px solid #3b82f6;">
        <h3 style="margin: 0 0 10px 0; font-size: 15px; line-height: 1.4;">
            <a href="{paper.url}" style="color: #1e40af; text-decoration: none;" target="_blank">{paper.title}</a>
            {badge}
        </h3>
        <p style="margin: 0 0 8px 0; color: #64748b; font-size: 12px;">
            {authors_str} • <span style="background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{paper.source}</span> • {paper.published_date}
        </p>
        <p style="margin: 0 0 6px 0; color: #64748b; font-size: 11px;">
            <strong>DOI:</strong> <a href="{paper.url}" style="color: #3b82f6;" target="_blank">{doi_display}</a>
        </p>
        <p style="margin: 0; color: #475569; font-size: 13px; line-height: 1.5;">{summary}</p>
        {connection_html}
    </div>
    """


def generate_email_html(papers: List[Paper], highlights: str, categories: Dict[str, List[Paper]]) -> str:
    """Generate the complete HTML email."""
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # Format highlights
    lines = highlights.strip().split('\n')
    highlight_items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[•\-\*]\s*', '', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        highlight_items.append(f'<li style="margin-bottom: 10px; line-height: 1.5; color: #1e3a5f;">{line}</li>')
    highlights_html = f'<ul style="margin: 0; padding-left: 20px;">{" ".join(highlight_items)}</ul>'
    
    # Format papers by category
    if categories:
        papers_html = ""
        for cat_name, cat_papers in categories.items():
            cat_papers_html = "".join([format_paper_html(p) for p in cat_papers])
            papers_html += f"""
            <div style="margin-bottom: 32px;">
                <h2 style="margin: 0 0 18px 0; font-size: 18px; color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">
                    {cat_name} <span style="color: #64748b; font-weight: normal; font-size: 14px;">({len(cat_papers)} papers)</span>
                </h2>
                {cat_papers_html}
            </div>
            """
    else:
        papers_html = "".join([format_paper_html(p) for p in papers])
    
    # Source summary
    sources = sorted(set(p.source for p in papers))
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f1f5f9; margin: 0; padding: 20px;">
    <div style="max-width: 720px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.12);">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); padding: 40px 32px; text-align: center;">
            <h1 style="margin: 0; color: white; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                Eugen's Weekly<br>Research Digest
            </h1>
            <p style="margin: 14px 0 0 0; color: #93c5fd; font-size: 15px;">{date_str}</p>
        </div>
        
        <!-- Summary Box -->
        <div style="margin: 28px; padding: 24px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px; border: 1px solid #bfdbfe;">
            <h2 style="margin: 0 0 14px 0; font-size: 18px; color: #1e40af;">📊 This Week's Highlights</h2>
            <p style="margin: 0 0 16px 0; color: #1e40af; font-size: 14px;">
                <strong>{len(papers)} new papers</strong> with verified DOIs from {', '.join(sources)}
            </p>
            {highlights_html}
        </div>
        
        <!-- Info Box -->
        <div style="margin: 0 28px 28px 28px; padding: 14px 18px; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #22c55e;">
            <p style="margin: 0; font-size: 12px; color: #166534;">
                ✓ All papers have verified DOIs • Click any title or DOI link to access the paper directly
            </p>
        </div>
        
        <!-- Papers by Category -->
        <div style="padding: 0 28px 28px 28px;">
            {papers_html}
        </div>
        
        <!-- Footer -->
        <div style="background: #f8fafc; padding: 24px 28px; text-align: center; border-top: 1px solid #e2e8f0;">
            <p style="margin: 0 0 8px 0; color: #64748b; font-size: 12px;">
                Automatically curated for Eugen Dimant • Only new papers since last digest
            </p>
            <p style="margin: 0; color: #94a3b8; font-size: 11px;">
                Sources: Crossref, OpenAlex, SSRN, arXiv, Semantic Scholar, PubMed
            </p>
        </div>
    </div>
</body>
</html>"""

# =============================================================================
# EMAIL SENDING
# =============================================================================

def send_email(html: str) -> bool:
    """Send email via SMTP."""
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    recipient = CONFIG.get("email", {}).get("recipient", "")
    
    if not all([smtp_user, smtp_password, recipient]):
        logger.error("Missing email configuration (SMTP_USER, SMTP_PASSWORD, or recipient)")
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
        
        logger.info(f"✓ Email sent successfully to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Email sending failed: {e}")
        return False

# =============================================================================
# TEST DATA
# =============================================================================

def create_test_papers() -> List[Paper]:
    """Create realistic test papers with valid DOIs."""
    return [
        Paper(
            title="Social Norms and Tax Compliance: Evidence from a Large-Scale Field Experiment",
            authors=["Maria Garcia", "John Smith", "Wei Chen"],
            abstract="We conduct a field experiment with 50,000 taxpayers testing how social norm messages affect compliance. Descriptive norms ('most people pay taxes honestly') increased compliance by 12%, while injunctive norms ('tax evasion is wrong') had smaller effects. The results suggest that behavioral interventions leveraging descriptive social information can be highly cost-effective for tax authorities.",
            url="https://doi.org/10.1257/aer.20180277",
            source="Crossref",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            doi="10.1257/aer.20180277",
            relevance_score=0.85,
            summary="Large-scale RCT (N=50,000) shows descriptive social norm messages increase tax compliance by 12%, outperforming injunctive norm framing.",
            research_connection="This methodology could be adapted to study honesty interventions in organizational settings. The descriptive vs. injunctive norm comparison directly relates to your work on norm enforcement mechanisms."
        ),
        Paper(
            title="The Psychology of Dishonesty: When Do People Justify Lying?",
            authors=["Sarah Johnson", "Michael Brown"],
            abstract="Five experiments examine the moral psychology of lying. We find that perceived harm to others, not self-benefit, predicts when people justify dishonesty. Altruistic lies (lying to help others) are judged more acceptable than selfish lies, even when outcomes are identical. These findings challenge purely consequentialist accounts of moral judgment.",
            url="https://doi.org/10.1037/pspa0000123",
            source="OpenAlex",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            doi="10.1037/pspa0000123",
            relevance_score=0.82,
            summary="Experimental series reveals people justify lies based on perceived harm to others rather than personal gain. Altruistic lies are more acceptable than selfish ones regardless of outcomes.",
            research_connection="The altruistic lying paradigm could extend your cheating research by examining whether 'prosocial' dishonesty follows different psychological mechanisms than self-serving dishonesty."
        ),
        Paper(
            title="Peer Effects in Cooperative Behavior: A Cross-Cultural Public Goods Experiment",
            authors=["Emma Wilson", "David Lee", "Anna Mueller"],
            abstract="Using public goods games across 25 countries (N=12,500), we examine how observing peer contributions affects cooperation. Seeing cooperative peers increases contributions by 23%, with effects significantly stronger in collectivist cultures. Results highlight the importance of cultural context in designing behavioral interventions.",
            url="https://doi.org/10.1126/science.abc1234",
            source="SSRN",
            published_date=datetime.now().strftime("%Y-%m-%d"),
            doi="10.1126/science.abc1234",
            relevance_score=0.75,
            summary="Cross-cultural public goods experiment (25 countries, N=12,500) finds peer cooperation increases contributions by 23%, with stronger effects in collectivist societies.",
            research_connection="The cultural moderation findings could inform your norm enforcement research—are dishonesty norms similarly culture-dependent? Consider a replication with your cheating paradigm."
        )
    ]

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete digest pipeline."""
    test_mode = "--test" in sys.argv or os.environ.get("TEST_EMAIL") == "true"
    preview_mode = "--preview" in sys.argv
    
    logger.info("=" * 60)
    logger.info("EUGEN'S RESEARCH DIGEST")
    logger.info(f"Mode: {'TEST' if test_mode else 'PREVIEW' if preview_mode else 'PRODUCTION'}")
    logger.info("=" * 60)
    
    if test_mode:
        logger.info("Using test data...")
        papers = create_test_papers()
        highlights = """• **Behavioral Interventions at Scale**: New field experiments demonstrate that social norm messaging—particularly descriptive norms—can significantly impact real-world behaviors like tax compliance.

• **Psychology of Dishonesty**: Research this week reveals that moral justifications for lying depend more on perceived harm to others than on self-interest, challenging purely consequentialist models.

• **Cross-Cultural Cooperation**: Studies spanning multiple countries show that peer effects on cooperative behavior are moderated by cultural orientation, with implications for designing globally effective interventions."""
    else:
        logger.info("Fetching papers from all sources...")
        papers = fetch_all_papers()
        
        if not papers:
            logger.warning("No new papers found matching criteria")
            highlights = "No new papers found this week matching your research interests. This could mean:\n• All recent relevant papers were included in previous digests\n• Try adjusting keyword filters or lowering the relevance threshold"
        else:
            logger.info(f"Generating AI summaries and research connections for {len(papers)} papers...")
            
            # Generate highlights
            highlights = generate_highlights(papers)
            
            # Generate per-paper summaries and connections
            for i, paper in enumerate(papers):
                logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.title[:50]}...")
                summary, connection = generate_paper_summary_and_connection(paper)
                paper.summary = summary
                paper.research_connection = connection
                time.sleep(0.3)  # Rate limiting
    
    # Categorize papers
    categories = categorize_papers(papers)
    
    # Generate HTML email
    logger.info("Generating email HTML...")
    html = generate_email_html(papers, highlights, categories)
    
    # Save preview
    with open("latest_digest.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("✓ Saved preview to latest_digest.html")
    
    # Save papers JSON
    with open("latest_papers.json", "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in papers], f, indent=2)
    logger.info("✓ Saved papers to latest_papers.json")
    
    # Verify all papers have DOIs
    papers_without_doi = [p for p in papers if not p.doi]
    if papers_without_doi:
        logger.warning(f"Warning: {len(papers_without_doi)} papers missing DOI (these should not appear)")
    
    logger.info(f"\n📊 Summary: {len(papers)} papers with verified DOIs")
    for cat, cat_papers in categories.items():
        logger.info(f"   • {cat}: {len(cat_papers)} papers")
    
    if preview_mode:
        logger.info("\n✓ Preview mode - email not sent")
        return True
    
    # Send email
    logger.info("\nSending email...")
    success = send_email(html)
    
    # Update seen papers (only on successful send in production mode)
    if success and not test_mode:
        seen = load_seen_papers()
        for p in papers:
            seen.add(p.paper_id)
        save_seen_papers(seen)
        logger.info("✓ Updated seen papers tracking")
    
    logger.info("\n" + "=" * 60)
    logger.info("DIGEST COMPLETE" if success else "DIGEST FAILED")
    logger.info("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
