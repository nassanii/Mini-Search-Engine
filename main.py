"""
Main entry point for the Mini Search Engine.
"""

import argparse
import sys
import os

from src.core.preprocessing import TextPreprocessor
from src.core.incidence_matrix import IncidenceMatrix
from src.core.inverted_index import InvertedIndex
from src.core.positional_index import PositionalIndex
from src.core.ranking import Ranker
from src.features.spell_checker import SpellChecker
from src.features.query_expansion import QueryExpander
from src.features.snippet_generator import SnippetGenerator

def main() -> None:
    """
    Main function to run the search engine CLI.
    """
    parser = argparse.ArgumentParser(description="Mini Search Engine")
    parser.add_argument('query', type=str, nargs='?', help="The search query")
    parser.add_argument('--dir', type=str, default='data/documents', help="Directory containing documents")

    args = parser.parse_args()

    # If no query is provided, print help and exit
    if not args.query:
        parser.print_help()
        sys.exit(0)

    print(f"Searching for: '{args.query}' in directory '{args.dir}'")

    # 1. Initialize components
    preprocessor = TextPreprocessor(method='lemmatization')
    index = InvertedIndex()
    pos_index = PositionalIndex()
    inc_matrix = IncidenceMatrix()
    ranker = Ranker()
    spell_checker = SpellChecker()
    query_expander = QueryExpander()
    snippet_gen = SnippetGenerator(documents_dir=args.dir)

    # 2. Load and preprocess documents
    documents: dict[str, list[str]] = {}
    if os.path.exists(args.dir):
        for filename in os.listdir(args.dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(args.dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        tokens = preprocessor.process(text)
                        documents[filename] = tokens
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    else:
        print(f"Directory {args.dir} not found. Creating it...")
        os.makedirs(args.dir, exist_ok=True)
        print("Please add some .txt files to the directory and try again.")
        sys.exit(0)

    if not documents:
        print("No documents found to index.")
        sys.exit(0)

    # 3. Build Indices
    print(f"Building indices for {len(documents)} documents...")
    index.build_index(documents)
    pos_index.build_index(documents)
    inc_matrix.build_index(documents)

    # 4. Rank Documents
    print(f"Computing TF-IDF for {len(documents)} documents...")
    ranker.compute_tf_idf(documents)
    
    # Load vocabulary into spell checker
    spell_checker.load_vocabulary(ranker.vocabulary)
    
    # 5. Process Query
    raw_tokens = args.query.split()
    corrections = spell_checker.get_corrections(raw_tokens)
    if corrections:
        corrected_query = " ".join([corrections.get(w, w) for w in raw_tokens])
        print(f"\n[Spelling Correction] Did you mean: '{corrected_query}'?")
    
    query_tokens = preprocessor.process(args.query)
    
    print("\n[Query Expansion] Expanding query...")
    expanded_tokens = query_expander.expand_query(query_tokens)
    print(f"Original tokens: {query_tokens}")
    print(f"Expanded tokens: {expanded_tokens}")
    
    search_tokens = expanded_tokens
    
    # --- Print Data Structures for the Query Terms ---
    print("\n" + "="*60)
    print("DATA STRUCTURES FOR QUERY TERMS ")
    print("="*60)
    
    # 1. Incidence Matrix Representation
    print("\n[1] Incidence Matrix (Showing up to 10 documents containing the terms):")
    if inc_matrix.df is not None:
        valid_terms = [t for t in search_tokens if t in inc_matrix.df.index]
        if valid_terms:
            sub_df = inc_matrix.df.loc[valid_terms]
            # Get columns where ALL terms exist (Boolean AND matches)
            all_match_docs = sub_df.columns[(sub_df == 1).all(axis=0)].tolist()
            
            # Get columns where ANY term exists (Partial matches)
            any_match_docs = [doc for doc in sub_df.columns[(sub_df == 1).any(axis=0)] if doc not in all_match_docs]
            
            active_docs = all_match_docs + any_match_docs
            
            if len(active_docs) > 0:
                # Show up to 10 documents clearly, prioritizing those that match BOTH words
                display_docs = active_docs[:10]
                print(sub_df[display_docs].to_string())
                if len(active_docs) > 10:
                    print(f"  ... and {len(active_docs) - 10} more matching documents.")
            else:
                print("Terms exist in vocabulary, but no matching documents.")
        else:
            print("Terms not found in Incidence Matrix vocabulary.")
            
    # 2. Inverted Index Representation
    print("\n[2] Inverted Index (Posting Lists):")
    for term in search_tokens:
        posting_list = index.index.get(term, set())
        print(f"'{term}' -> {list(posting_list)[:5]} ... (total docs: {len(posting_list)})")
        
    # 3. Positional Index Representation
    print("\n[3] Positional Index (Positions mapping):")
    for term in search_tokens:
        pos_list = pos_index.index.get(term, {})
        sample_pos = dict(list(pos_list.items())[:3]) # Show first 3 docs
        print(f"'{term}' -> {sample_pos} ... (total docs: {len(pos_list)})")

    print("\n" + "="*60)
    print(" SEARCH ENGINE RESULTS ")
    print("="*60)
    
    # 5. Execute searches across all models
    
    # A. Incidence Matrix Search
    inc_matched_docs = inc_matrix.search(search_tokens)
    print(f"\n1. Results based on Incidence Matrix (Boolean AND):")
    if inc_matched_docs:
        for doc in inc_matched_docs[:10]:
            print(f"- {doc}")
    else:
        print("No matches found.")
        
    # B. Inverted Index Search
    inv_matched_docs = index.search(search_tokens)
    print(f"\n2. Results based on Inverted Index (Boolean AND):")
    if inv_matched_docs:
        for doc in inv_matched_docs[:10]:
            print(f"- {doc}")
    else:
        print("No matches found.")
    
    # C. Exact Phrase Search (Positional Index)
    phrase_matched_docs = pos_index.search_phrase(search_tokens)
    print(f"\n3. Results based on Exact Phrase Search ('{args.query}'):")
    if phrase_matched_docs:
        for doc in phrase_matched_docs[:10]:
            print(f"- {doc}")
    else:
        print("No exact phrase matches found.")
        
    # D. Ranked Retrieval
    ranked_docs = ranker.rank_documents(search_tokens)
    print(f"\n4. Results based on Ranked Retrieval (TF-IDF & Cosine Similarity):")
    if ranked_docs:
        # Show top 10 results
        for rank, (doc, score) in enumerate(ranked_docs[:10], start=1):
            snippet = snippet_gen.get_snippet(doc, search_tokens)
            print(f"{rank}. {doc} (Similarity Score: {score:.4f})")
            print(f"   Snippet: ...{snippet}...\n")
            
    else:
        print("No matching documents found.")


if __name__ == '__main__':
    main()
