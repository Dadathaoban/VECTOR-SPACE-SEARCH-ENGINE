#!/usr/bin/env python3
"""
IR System Main Entry Point
Usage:
    python main.py index <corpus_path> [--output <dir>]
    python main.py search [--index <dir>] [--query <query>]
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='IR System - Index and Search')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build inverted index')
    index_parser.add_argument('corpus_path', help='Path to corpus directory or file')
    index_parser.add_argument('--output', '-o', default='.', help='Output directory for index files')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('--index', '-i', default='.', help='Directory containing index files')
    search_parser.add_argument('--query', '-q', help='Query to search')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        from indexer import main as indexer_main
        sys.argv = ['indexer.py', args.corpus_path, '--output', args.output]
        indexer_main()
    
    elif args.command == 'search':
        from search import main as search_main
        if args.query:
            sys.argv = ['search.py', '--index', args.index, '--query', args.query]
        else:
            sys.argv = ['search.py', '--index', args.index]
        search_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()