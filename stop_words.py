#!/usr/bin/env python3
"""
Extended Stop Words List for Web Crawler
Includes comprehensive list for better filtering
"""

# Comprehensive stop words list
STOP_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Common conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet', 'for',
    
    # Prepositions
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
    'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over',
    'past', 'since', 'through', 'throughout', 'till', 'to', 'toward', 'under',
    'underneath', 'until', 'up', 'upon', 'with', 'within', 'without',
    
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    
    # Common verbs (to be, to have, etc.)
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    
    # Common adverbs
    'very', 'too', 'quite', 'rather', 'somewhat', 'almost', 'just', 'only',
    'really', 'even', 'ever', 'never', 'always', 'often', 'sometimes', 'usually',
    'well', 'better', 'best', 'bad', 'worse', 'worst',
    
    # Common adjectives
    'all', 'any', 'both', 'each', 'every', 'few', 'many', 'more', 'most',
    'much', 'several', 'some', 'such', 'no', 'none', 'other', 'same',
    'different', 'new', 'old', 'good', 'great', 'high', 'small', 'large',
    'big', 'long', 'little', 'young', 'important', 'early', 'late',
    
    # Common contractions
    "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't",
    "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
    
    # Common connectors
    'also', 'however', 'therefore', 'thus', 'hence', 'consequently',
    'furthermore', 'moreover', 'nevertheless', 'nonetheless', 'otherwise',
    'similarly', 'accordingly', 'besides', 'else', 'instead', 'likewise',
    'meanwhile', 'moreover', 'namely', 'next', 'now', 'otherwise', 'still',
    'then', 'thereafter', 'therefore', 'thus', 'undoubtedly',
    
    # Web-specific common words
    'click', 'here', 'home', 'page', 'website', 'web', 'site', 'link',
    'menu', 'navigation', 'footer', 'header', 'sidebar', 'content',
    'copyright', 'privacy', 'policy', 'terms', 'conditions', 'contact',
    'about', 'us', 'our', 'services', 'products', 'blog', 'news',
    'read', 'more', 'view', 'download', 'subscribe', 'sign', 'up',
    'login', 'logout', 'register', 'account', 'profile', 'settings',
    
    # Time-related
    'today', 'tomorrow', 'yesterday', 'now', 'then', 'when', 'while',
    'before', 'after', 'during', 'since', 'until', 'soon', 'later',
    'early', 'late', 'always', 'never', 'often', 'sometimes', 'usually',
    
    # Place-related
    'here', 'there', 'where', 'everywhere', 'anywhere', 'somewhere',
    'above', 'below', 'under', 'over', 'between', 'among', 'through',
    
    # Question words
    'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'when', 'where',
    
    # Demonstratives
    'this', 'that', 'these', 'those',
    
    # Numbers (common ones)
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next', 'previous',
    
    # Common web crawler noise
    'com', 'www', 'http', 'https', 'html', 'htm', 'php', 'asp', 'jsp',
    'index', 'default', 'main', 'homepage'
}

def is_stop_word(word):
    """Check if a word is a stop word."""
    return word.lower() in STOP_WORDS

def add_stop_word(word):
    """Add a word to the stop words list."""
    STOP_WORDS.add(word.lower())

def remove_stop_word(word):
    """Remove a word from the stop words list."""
    STOP_WORDS.discard(word.lower())

def get_stop_words():
    """Get the complete stop words list."""
    return sorted(list(STOP_WORDS))

def load_custom_stop_words(file_path):
    """Load custom stop words from a file."""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    STOP_WORDS.add(word)
        print(f"Loaded {len(STOP_WORDS)} stop words from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading stop words: {e}")
        return False

def save_stop_words(file_path):
    """Save stop words to a file."""
    try:
        with open(file_path, 'w') as f:
            for word in sorted(STOP_WORDS):
                f.write(f"{word}\n")
        print(f"Saved {len(STOP_WORDS)} stop words to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving stop words: {e}")
        return False

# Test function
def test_stop_words():
    """Test the stop words functionality."""
    test_words = ['the', 'computer', 'algorithm', 'and', 'data', 'structure']
    
    print("Testing stop words:")
    for word in test_words:
        if is_stop_word(word):
            print(f"  '{word}' is a stop word")
        else:
            print(f"  '{word}' is NOT a stop word")
    
    print(f"\nTotal stop words in list: {len(STOP_WORDS)}")
    print(f"Sample stop words: {sorted(list(STOP_WORDS))[:20]}...")

if __name__ == "__main__":
    test_stop_words()