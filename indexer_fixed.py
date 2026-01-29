# Save this as indexer_fixed.py
import os
import sys
import math
from collections import defaultdict

# Add the utility imports
sys.path.append('.')
from utils import *
from stop_words import is_stop_word
from porterstemmer import PorterStemmer

class IndexerFixed:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.stemmer = PorterStemmer()
        
        # Data structures
        self.documents = {}
        self.terms = {}
        self.postings = defaultdict(dict)
        self.doc_lengths = {}
        
        # Counters
        self.next_doc_id = 1
        self.next_term_id = 1
        
        # Statistics
        self.total_docs = 0
        self.total_terms = 0
    
    def build_index(self):
        """Build index from any text files"""
        print(f"Building index from: {self.corpus_path}")
        
        # Get all files
        if os.path.isfile(self.corpus_path):
            # Single file - might contain multiple documents
            doc_files = [self.corpus_path]
        elif os.path.isdir(self.corpus_path):
            # Directory - get all text-like files
            doc_files = []
            for root, dirs, files in os.walk(self.corpus_path):
                for file in files:
                    # Accept more file types
                    if any(file.endswith(ext) for ext in ['.txt', '.cacm', '.all', '.html', '.xml', '']):
                        doc_files.append(os.path.join(root, file))
        else:
            print(f"Path not found: {self.corpus_path}")
            return False
        
        print(f"Found {len(doc_files)} files to process")
        
        if not doc_files:
            print("No files found! Check your corpus path.")
            return False
        
        # Process each file
        processed_count = 0
        for i, doc_file in enumerate(doc_files, 1):
            print(f"Processing file {i}/{len(doc_files)}: {os.path.basename(doc_file)}")
            
            try:
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # If it's a large file, it might contain multiple documents
                # For CACM, documents might be separated by markers
                if len(content) > 100000:  # Large file
                    print(f"  Large file detected ({len(content)} chars), checking for document markers...")
                    
                    # Try different CACM document separators
                    separators = ['.I ', '.T\n', '.W\n', '.B\n', '.A\n', '.N\n', '.X\n']
                    
                    for sep in separators:
                        if sep in content:
                            print(f"  Found separator '{sep}', splitting documents...")
                            docs = content.split(sep)
                            for doc_num, doc_content in enumerate(docs[1:], 1):  # Skip first
                                if doc_content.strip():
                                    doc_id = self.process_document_content(doc_content, f"{doc_file}#doc{doc_num}")
                                    if doc_id:
                                        processed_count += 1
                            break
                    else:
                        # No separators found, treat as single document
                        doc_id = self.process_document_content(content, doc_file)
                        if doc_id:
                            processed_count += 1
                else:
                    # Small file, treat as single document
                    doc_id = self.process_document_content(content, doc_file)
                    if doc_id:
                        processed_count += 1
                        
            except Exception as e:
                print(f"  Error processing {doc_file}: {e}")
        
        if processed_count == 0:
            print("No documents were processed!")
            return False
        
        # Calculate weights
        print(f"\nCalculating TF-IDF for {processed_count} documents...")
        self.calculate_tf_idf()
        
        # Calculate document lengths
        print("Calculating document vector lengths...")
        self.calculate_document_lengths()
        
        # Update statistics
        self.total_docs = len(self.documents)
        self.total_terms = len(self.terms)
        
        print(f"\nIndex built successfully!")
        print(f"Total documents: {self.total_docs}")
        print(f"Total unique terms: {self.total_terms}")
        
        return True
    
    def process_document_content(self, content, doc_path):
        """Process document content and extract terms."""
        try:
            doc_id = self.next_doc_id
            self.documents[doc_path] = doc_id
            self.next_doc_id += 1
            
            # Tokenize content
            tokens = splitchars(content)
            term_freq = defaultdict(int)
            
            for token in tokens:
                # Normalize and filter token
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
                
                # Update term frequency
                term_freq[stemmed] += 1
            
            # Update postings
            for term, freq in term_freq.items():
                if term not in self.terms:
                    self.terms[term] = self.next_term_id
                    self.next_term_id += 1
                
                term_id = self.terms[term]
                self.postings[term_id][doc_id] = {'tf': freq}
            
            return doc_id
            
        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")
            return None
    
    def calculate_tf_idf(self):
        """Calculate TF-IDF weights."""
        N = len(self.documents)
        
        for term_id, doc_postings in self.postings.items():
            df = len(doc_postings)
            
            if df == 0:
                continue
                
            idf = math.log(N / df)
            
            for doc_id, posting in doc_postings.items():
                tf = posting['tf']
                tf_idf = tf * idf
                posting['tf_idf'] = tf_idf
                posting['idf'] = idf
                posting['df'] = df
    
    def calculate_document_lengths(self):
        """Calculate document vector lengths."""
        for doc_path, doc_id in self.documents.items():
            doc_vector_length = 0
            
            for term_id, doc_postings in self.postings.items():
                if doc_id in doc_postings:
                    tf_idf = doc_postings[doc_id]['tf_idf']
                    doc_vector_length += tf_idf * tf_idf
            
            self.doc_lengths[doc_id] = math.sqrt(doc_vector_length)
    
    def save_index(self, output_dir='.'):
        """Save index to files."""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save documents
            with open(os.path.join(output_dir, 'documents.dat'), 'w') as f:
                for doc_path, doc_id in self.documents.items():
                    f.write(f"{doc_path},{doc_id}\n")
            
            # Save terms
            with open(os.path.join(output_dir, 'terms.dat'), 'w') as f:
                for term, term_id in self.terms.items():
                    f.write(f"{term},{term_id}\n")
            
            # Save postings
            with open(os.path.join(output_dir, 'postings.dat'), 'w') as f:
                for term_id, doc_postings in self.postings.items():
                    for doc_id, posting in doc_postings.items():
                        tf_idf = posting.get('tf_idf', 0)
                        tf = posting.get('tf', 0)
                        idf = posting.get('idf', 0)
                        df = posting.get('df', 0)
                        f.write(f"{term_id},{doc_id},{tf_idf:.6f},{tf},{idf:.6f},{df}\n")
            
            # Save document lengths
            with open(os.path.join(output_dir, 'doc_lengths.dat'), 'w') as f:
                for doc_id, length in self.doc_lengths.items():
                    f.write(f"{doc_id},{length:.6f}\n")
            
            print(f"Index saved to {output_dir}/")
            return True
            
        except Exception as e:
            print(f"Error saving index: {e}")
            return False

# Quick test
if __name__ == "__main__":
    cacm_path = r"C:\Users\dell\Downloads\cacm"
    indexer = IndexerFixed(cacm_path)
    if indexer.build_index():
        indexer.save_index("./index_data")