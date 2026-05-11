"""
Positional Index implementation.
"""

from src.core.base_index import Index
import fnmatch

class PositionalIndex(Index):
    """
    Positional Index implementation for phrase queries.
    """

    def __init__(self) -> None:
        """Initializes an empty Positional Index."""
        # Maps a term to a dictionary mapping document IDs to a list of positions
        self.index: dict[str, dict[str, list[int]]] = {}

    def build_index(self, documents: dict[str, list[str]]) -> None:
        """
        Builds the Positional Index.

        Args:
            documents (dict[str, list[str]]): Documents to index.
        """
        for doc_id, terms in documents.items():
            for position, term in enumerate(terms):
                if term not in self.index:
                    self.index[term] = {}
                if doc_id not in self.index[term]:
                    self.index[term][doc_id] = []
                self.index[term][doc_id].append(position)

    def get_wildcard_matches(self, pattern: str) -> list[str]:
        """Finds all vocabulary terms matching a wildcard pattern."""
        return fnmatch.filter(self.index.keys(), pattern)

    def search_phrase(self, phrase_query: list[str]) -> list[str]:
        """
        Searches the Positional Index for exact consecutive phrases.
        Supports wildcards by expanding them, but combinatorial expansion can be expensive.
        We will check if any combination of the expanded terms forms a phrase.

        Args:
            phrase_query (list[str]): Ordered search terms.

        Returns:
            list[str]: Matching document IDs.
        """
        if not phrase_query:
            return []
            
        # 1. Expand each term (if it's a wildcard, it becomes a list of terms; otherwise, a list of one term)
        expanded_query: list[list[str]] = []
        for term in phrase_query:
            if '*' in term or '?' in term:
                matches = self.get_wildcard_matches(term)
                if not matches:
                    return [] # Wildcard matched nothing
                expanded_query.append(matches)
            else:
                if term not in self.index:
                    return []
                expanded_query.append([term])
                
        # 2. Find common documents that contain at least one variant of each position
        common_docs = None
        for variants in expanded_query:
            docs_for_position = set()
            for variant in variants:
                docs_for_position.update(self.index[variant].keys())
            if common_docs is None:
                common_docs = docs_for_position
            else:
                common_docs = common_docs.intersection(docs_for_position)
            
        if not common_docs:
            return []
            
        matching_docs: list[str] = []
        
        # 3. For each common document, verify if terms are consecutive
        for doc_id in common_docs:
            # We need to find if there is ANY valid phrase sequence in this doc.
            # Get positions for the first term variants
            first_term_positions = []
            for variant in expanded_query[0]:
                if doc_id in self.index[variant]:
                    first_term_positions.extend(self.index[variant][doc_id])
            
            for pos in first_term_positions:
                is_phrase_match = True
                
                for offset, variants in enumerate(expanded_query[1:], start=1):
                    expected_pos = pos + offset
                    # Check if ANY of the variants exist at the expected_pos
                    found_variant_at_pos = False
                    for variant in variants:
                        if doc_id in self.index[variant] and expected_pos in self.index[variant][doc_id]:
                            found_variant_at_pos = True
                            break
                    
                    if not found_variant_at_pos:
                        is_phrase_match = False
                        break
                        
                if is_phrase_match:
                    matching_docs.append(doc_id)
                    break  # Phrase found in this doc, no need to check further positions
                    
        return matching_docs

    def search(self, query_terms: list[str]) -> list[str]:
        """
        Overrides the base search method to use phrase search logic.
        """
        return self.search_phrase(query_terms)
