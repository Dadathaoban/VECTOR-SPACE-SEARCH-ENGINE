#!/usr/bin/env python3
"""
Enhanced Utilities for Web Crawler and IR System
Includes URL validation and HTML processing helpers
"""

import re
import string
import math
import urllib.parse

def is_number(s):
    """Check if a string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        try:
            int(s)
            return True
        except ValueError:
            return False

def is_short(s, limit=2):
    """Check if a string is shorter than or equal to limit."""
    return len(s) <= limit

def starts_with_punctuation(s):
    """Check if a string starts with punctuation."""
    if not s:
        return False
    return s[0] in string.punctuation

def splitchars(s):
    """Split a string into tokens based on non-alphanumeric characters."""
    return re.findall(r'[a-zA-Z0-9]+', s)

def normalize_token(token):
    """Normalize a token: lowercase and remove leading/trailing whitespace."""
    return token.lower().strip()

def compute_tf(term_freq):
    """Compute term frequency."""
    return term_freq

def compute_idf(N, df):
    """Compute inverse document frequency."""
    if df == 0:
        return 0
    return math.log(N / df)

def compute_tf_idf(tf, idf):
    """Compute TF-IDF weight."""
    return tf * idf

def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors."""
    dot_product = 0
    norm1 = 0
    norm2 = 0
    
    # Handle different vector representations
    if isinstance(vector1, dict) and isinstance(vector2, dict):
        # Both are sparse vectors
        for term_id in set(vector1.keys()).union(set(vector2.keys())):
            val1 = vector1.get(term_id, 0)
            val2 = vector2.get(term_id, 0)
            dot_product += val1 * val2
            norm1 += val1 * val1
            norm2 += val2 * val2
    elif isinstance(vector1, list) and isinstance(vector2, list):
        # Both are dense vectors
        for i in range(len(vector1)):
            dot_product += vector1[i] * vector2[i]
            norm1 += vector1[i] * vector1[i]
            norm2 += vector2[i] * vector2[i]
    else:
        # Mixed representation or other
        return 0
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

# NEW FUNCTIONS FOR WEB CRAWLER
def is_valid_url(url):
    """Validate URL format."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def normalize_url(url):
    """Normalize URL by removing fragments and trailing slashes."""
    try:
        parsed = urllib.parse.urlparse(url)
        
        # Remove fragment
        parsed = parsed._replace(fragment='')
        
        # Remove trailing slash from path
        path = parsed.path
        if path.endswith('/'):
            path = path.rstrip('/')
        parsed = parsed._replace(path=path)
        
        # Reconstruct URL
        return urllib.parse.urlunparse(parsed)
    except:
        return url

def get_domain_from_url(url):
    """Extract domain from URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc
    except:
        return ""

def should_skip_url(url):
    """Check if URL should be skipped (files, anchors, etc.)."""
    skip_extensions = [
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.exe', '.dmg', '.msi', '.iso',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
    ]
    
    url_lower = url.lower()
    
    # Skip anchors, javascript, mailto, etc.
    if url_lower.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'ftp:')):
        return True
    
    # Skip file extensions
    if any(url_lower.endswith(ext) for ext in skip_extensions):
        return True
    
    return False

def extract_links_from_html(html_content, base_url):
    """Extract all links from HTML content."""
    links = []
    
    # Simple regex to find href attributes
    href_pattern = r'href=[\'"]?([^\'" >]+)'
    src_pattern = r'src=[\'"]?([^\'" >]+)'
    
    # Find href links
    for match in re.findall(href_pattern, html_content, re.IGNORECASE):
        if not should_skip_url(match):
            absolute_url = urllib.parse.urljoin(base_url, match)
            if is_valid_url(absolute_url):
                links.append(absolute_url)
    
    # Find src links (images, scripts, etc.)
    for match in re.findall(src_pattern, html_content, re.IGNORECASE):
        if not should_skip_url(match):
            absolute_url = urllib.parse.urljoin(base_url, match)
            if is_valid_url(absolute_url):
                links.append(absolute_url)
    
    return links

def clean_html_text(text):
    """Clean text extracted from HTML."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def format_time(seconds):
    """Format seconds into hours:minutes:seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print progress bar."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    # Print new line on completion
    if iteration == total: 
        print()