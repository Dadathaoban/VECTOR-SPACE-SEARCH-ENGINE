import os
import sys
import math
from collections import defaultdict
from utils import *
from stop_words import is_stop_word
from porterstemmer import PorterStemmer

class SearchEngine:
    def __init__(self, index_dir='.'):
        self.index_dir = index_dir
        self.stemmer = PorterStemmer()
        
        # Loaded data
        self.documents = {}  # doc_path -> doc_id
        self.documents_inv = {}  # doc_id -> doc_path
        self.terms = {}  # term -> term_id
        self.terms_inv = {}  # term_id -> term
        self.postings = defaultdict(dict)  # term_id -> {doc_id: posting_data}
        self.doc_lengths = {}  # doc_id -> document length
        
        # Statistics
        self.total_docs = 0
        self.total_terms = 0
        
        # Load the index
        self.load_index()
        
    def load_index(self):
        """Load the inverted index from files."""
        try:
            print("Loading index files...")
            
            # Load documents
            self.documents = {}
            self.documents_inv = {}
            with open(os.path.join(self.index_dir, 'documents.dat'), 'r') as f:
                for line in f:
                    doc_path, doc_id = line.strip().split(',')
                    doc_id = int(doc_id)
                    self.documents[doc_path] = doc_id
                    self.documents_inv[doc_id] = doc_path
            
            # Load terms
            self.terms = {}
            self.terms_inv = {}
            with open(os.path.join(self.index_dir, 'terms.dat'), 'r') as f:
                for line in f:
                    term, term_id = line.strip().split(',')
                    term_id = int(term_id)
                    self.terms[term] = term_id
                    self.terms_inv[term_id] = term
            
            # Load postings
            self.postings = defaultdict(dict)
            with open(os.path.join(self.index_dir, 'postings.dat'), 'r') as f:
                for line in f:
                    term_id, doc_id, tf_idf, tf, idf, df = line.strip().split(',')
                    term_id = int(term_id)
                    doc_id = int(doc_id)
                    self.postings[term_id][doc_id] = {
                        'tf_idf': float(tf_idf),
                        'tf': int(tf),
                        'idf': float(idf),
                        'df': int(df)
                    }
            
            # Load document lengths
            self.doc_lengths = {}
            with open(os.path.join(self.index_dir, 'doc_lengths.dat'), 'r') as f:
                for line in f:
                    doc_id, length = line.strip().split(',')
                    self.doc_lengths[int(doc_id)] = float(length)
            
            # Update statistics
            self.total_docs = len(self.documents)
            self.total_terms = len(self.terms)
            
            print(f"Index loaded successfully!")
            print(f"Total documents: {self.total_docs}")
            print(f"Total unique terms: {self.total_terms}")
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def parse_query(self, query):
        """Parse and process the query terms."""
        parsed_terms = []
        tokens = splitchars(query)
        
        for token in tokens:
            # Normalize and filter
            token_lower = normalize_token(token)
            
            # Apply filters
            if is_stop_word(token_lower):
                continue
            if starts_with_punctuation(token_lower):
                continue
            if is_short(token_lower):
                continue
            if is_number(token_lower):
                continue
            
            # Stem the token
            stemmed = self.stemmer.stem(token_lower, 0, len(token_lower) - 1)
            
            # Check if term exists in index
            if stemmed in self.terms:
                term_id = self.terms[stemmed]
                parsed_terms.append({
                    'original': token,
                    'processed': stemmed,
                    'term_id': term_id
                })
            else:
                print(f"Term '{stemmed}' not found in index")
        
        return parsed_terms
    
    def build_query_vector(self, parsed_terms):
        """Build a query vector from parsed terms."""
        # Count term frequencies in query
        term_freq = defaultdict(int)
        for term_info in parsed_terms:
            term_freq[term_info['term_id']] += 1
        
        # Build sparse query vector with TF-IDF weights
        query_vector = {}
        N = self.total_docs
        
        for term_id, tf in term_freq.items():
            if term_id in self.postings:
                df = len(self.postings[term_id])
                idf = math.log(N / df) if df > 0 else 0
                query_vector[term_id] = tf * idf
        
        return query_vector
    
    def get_documents_with_all_terms(self, parsed_terms):
        """Find documents that contain ALL query terms."""
        if not parsed_terms:
            return set()
        
        # Start with documents containing the first term
        first_term_id = parsed_terms[0]['term_id']
        candidate_docs = set(self.postings[first_term_id].keys())
        
        # Intersect with documents containing other terms
        for term_info in parsed_terms[1:]:
            term_id = term_info['term_id']
            if term_id in self.postings:
                term_docs = set(self.postings[term_id].keys())
                candidate_docs = candidate_docs.intersection(term_docs)
            else:
                # If a term doesn't exist in any document, return empty set
                return set()
        
        return candidate_docs
    
    def search(self, query, top_k=20):
        """Search for documents relevant to the query."""
        print(f"\nSearching for: '{query}'")
        
        # Parse query
        parsed_terms = self.parse_query(query)
        
        if not parsed_terms:
            print("No valid query terms found!")
            return []
        
        print(f"Parsed terms: {[t['processed'] for t in parsed_terms]}")
        
        # Get documents containing all query terms
        candidate_docs = self.get_documents_with_all_terms(parsed_terms)
        
        if not candidate_docs:
            print("No documents contain all query terms!")
            return []
        
        print(f"Found {len(candidate_docs)} documents containing all query terms")
        
        # Build query vector
        query_vector = self.build_query_vector(parsed_terms)
        query_length = math.sqrt(sum(w * w for w in query_vector.values()))
        
        if query_length == 0:
            print("Query vector has zero length!")
            return []
        
        # Calculate cosine similarity for each candidate document
        results = []
        
        for doc_id in candidate_docs:
            # Get document vector (sparse)
            doc_vector = {}
            for term_id, weight in query_vector.items():
                if doc_id in self.postings[term_id]:
                    doc_vector[term_id] = self.postings[term_id][doc_id]['tf_idf']
            
            # Calculate dot product
            dot_product = 0
            for term_id, q_weight in query_vector.items():
                if term_id in doc_vector:
                    dot_product += q_weight * doc_vector[term_id]
            
            # Get document length
            doc_length = self.doc_lengths.get(doc_id, 1.0)
            
            # Calculate cosine similarity
            if doc_length > 0 and query_length > 0:
                similarity = dot_product / (doc_length * query_length)
            else:
                similarity = 0
            
            if similarity > 0:
                results.append({
                    'doc_id': doc_id,
                    'similarity': similarity,
                    'doc_path': self.documents_inv[doc_id]
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def print_results(self, results, query):
        """Print search results in a formatted way."""
        if not results:
            print("No results found!")
            return
        
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS FOR: '{query}'")
        print(f"Total documents retrieved: {len(results)}")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            doc_id = result['doc_id']
            similarity = result['similarity']
            doc_path = result['doc_path']
            doc_name = os.path.basename(doc_path)
            
            print(f"\n{i:3d}. Document: {doc_name}")
            print(f"     Document ID: {doc_id}")
            print(f"     Path: {doc_path}")
            print(f"     Cosine Similarity: {similarity:.6f}")
            
            # Show query terms in document
            parsed_terms = self.parse_query(query)
            term_weights = []
            for term_info in parsed_terms:
                term_id = term_info['term_id']
                if doc_id in self.postings[term_id]:
                    weight = self.postings[term_id][doc_id]['tf_idf']
                    term_weights.append(f"{term_info['processed']}:{weight:.3f}")
            
            if term_weights:
                print(f"     Term weights: {', '.join(term_weights)}")
        
        print(f"\n{'='*80}")
        print(f"Showing top {len(results)} results")
        print(f"{'='*80}\n")
    
    def interactive_search(self):
        """Run interactive search session."""
        print("\n" + "="*80)
        print("INTERACTIVE SEARCH ENGINE")
        print("="*80)
        print("Commands:")
        print("  - Enter a query to search")
        print("  - Type 'exit' to quit")
        print("  - Type 'clear' to clear screen")
        print("  - Type 'stats' to show index statistics")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("\nEnter query: ").strip()
                
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif query.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif query.lower() == 'stats':
                    print(f"\nIndex Statistics:")
                    print(f"  Total documents: {self.total_docs}")
                    print(f"  Total unique terms: {self.total_terms}")
                    print(f"  Average terms per document: {self.total_terms / self.total_docs:.1f}")
                    continue
                elif not query:
                    continue
                
                # Perform search
                results = self.search(query, top_k=20)
                
                # Print results
                self.print_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

def main():
    """Main function for the search engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search engine using inverted index')
    parser.add_argument('--index', '-i', default='.', help='Directory containing index files')
    parser.add_argument('--query', '-q', help='Query to search (optional, will run interactive if not provided)')
    
    args = parser.parse_args()
    
    # Create search engine
    engine = SearchEngine(args.index)
    
    if args.query:
        # Single query mode
        results = engine.search(args.query, top_k=20)
        engine.print_results(results, args.query)
    else:
        # Interactive mode
        engine.interactive_search()

if __name__ == "__main__":
    main()