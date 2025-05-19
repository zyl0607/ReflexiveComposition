# reflexive_composition/utils/text_utils.py
"""
Text analysis utilities for the Reflexive Composition framework.
"""

import re
from typing import Dict, List, Any, Optional, Union

def contains_temporal_info(text: str) -> bool:
    """
    Determine if a text contains significant temporal information.
    
    Args:
        text: Source text to analyze
        
    Returns:
        True if temporal information is detected, False otherwise
    """
    # Check for date patterns
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        r'\bon\s+\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December)'  # "on 13 July"
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for temporal keywords
    temporal_keywords = [
        'yesterday', 'today', 'tomorrow', 'next week', 'last week',
        'next month', 'last month', 'next year', 'last year',
        'recent', 'recently', 'latest', 'current', 'upcoming',
        'schedule', 'timeline', 'history', 'date', 'time', 
        'since', 'until', 'during', 'following'
    ]
    
    for keyword in temporal_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            return True
    
    return False


def identify_domain(text: str) -> Optional[str]:
    """
    Attempt to identify the domain of a text.
    
    Args:
        text: Source text to analyze
        
    Returns:
        Domain name if identified, None otherwise
    """
    # Medical domain indicators
    medical_terms = [
        'patient', 'diagnosis', 'treatment', 'symptom', 'prescription',
        'doctor', 'hospital', 'clinic', 'medication', 'disease',
        'medical', 'healthcare', 'nurse', 'physician', 'surgery'
    ]
    
    # Financial domain indicators
    financial_terms = [
        'investment', 'market', 'stock', 'bond', 'finance',
        'bank', 'currency', 'transaction', 'economic', 'fiscal',
        'revenue', 'profit', 'budget', 'portfolio', 'dividend'
    ]
    
    # News domain indicators
    news_terms = [
        'reported', 'announced', 'according to', 'press', 'media',
        'statement', 'official', 'spokesperson', 'release', 'interview',
        'breaking', 'headline', 'sources say', 'investigation', 'coverage'
    ]
    
    # Count domain indicators
    medical_count = sum(1 for term in medical_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    financial_count = sum(1 for term in financial_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    news_count = sum(1 for term in news_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    
    # Get the domain with the highest count
    domain_counts = {
        'medical': medical_count,
        'financial': financial_count,
        'news': news_count
    }
    
    # Get the domain with the highest count if it exceeds a threshold
    threshold = 3  # Minimum number of domain-specific terms needed
    max_domain = max(domain_counts.items(), key=lambda x: x[1])
    
    if max_domain[1] >= threshold:
        return max_domain[0]
    
    return None


def suggest_extraction_type(text: str) -> Dict[str, Any]:
    """
    Suggest an extraction type based on text analysis.
    
    Args:
        text: Source text to analyze
        
    Returns:
        Dictionary with extraction type and domain
    """
    is_temporal = contains_temporal_info(text)
    domain = identify_domain(text)
    
    if domain:
        return {
            'extraction_type': 'domain',
            'domain': domain
        }
    elif is_temporal:
        return {
            'extraction_type': 'temporal',
            'domain': None
        }
    else:
        return {
            'extraction_type': 'general',
            'domain': None
        }
