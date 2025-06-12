import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple, Union
from utils import timeit

logger = logging.getLogger(__name__)


@timeit
def verify_sentence(text: str, sentence: str) -> bool:
    """
    Verify if the sentence provided by the LLM is actually included in the original text.
    
    Args:
        text (str): Original report text
        sentence (str): Sentence provided by the LLM
    
    Returns:
        bool: Whether the sentence is included in the original text
    """
    if not sentence.strip():
        return False
    
    # Exact match
    if sentence in text:
        return True
    
    # Normalize whitespace and special characters for comparison
    clean_text = re.sub(r'\s+', ' ', text).strip()
    clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    if clean_sentence in clean_text:
        return True
    
    # Partial match (75% or higher similarity)
    similarity = calculate_similarity(clean_text, clean_sentence)
    if similarity >= 0.9:
        return True
        
    return False


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two strings.
    
    Args:
        text1 (str): First string
        text2 (str): Second string
    
    Returns:
        float: Similarity between the two strings (0.0~1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Calculate similarity using SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()