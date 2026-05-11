"""
Inverted Index implementation.
"""

from src.core.base_index import Index
import fnmatch

class InvertedIndex(Index):
    """
    Inverted Index implementation.
    """

    def __init__(self) -> None:
        """Initializes an empty Inverted Index."""
        # Maps a term to a set of document IDs
        self.index: dict[str, set[str]] = {}

    def build_index(self, documents: dict[str, list[str]]) -> None:
        """
        Builds the Inverted Index.

        Args:
            documents (dict[str, list[str]]): Documents to index.
        """
        for doc_id, terms in documents.items():
            for term in terms:
                if term not in self.index:
                    self.index[term] = set()
                self.index[term].add(doc_id)

    def get_wildcard_matches(self, pattern: str) -> list[str]:
        """
        Finds all vocabulary terms matching a wildcard pattern.

        Args:
            pattern (str): The wildcard pattern (e.g., 'comp*').

        Returns:
            list[str]: Matching terms.
        """
        return fnmatch.filter(self.index.keys(), pattern)

    def search(self, query_terms: list[str]) -> list[str]:
        """
        Performs a boolean AND search on the Inverted Index, supporting wildcards.

        Args:
            query_terms (list[str]): Search terms.

        Returns:
            list[str]: Matching document IDs.
        """
        if not query_terms:
            return []
        
        result_set = None
        
        for term in query_terms:
            current_term_set = set()
            if '*' in term or '?' in term:
                matches = self.get_wildcard_matches(term)
                for match in matches:
                    current_term_set.update(self.index.get(match, set()))
            else:
                current_term_set = self.index.get(term, set())
                
            if result_set is None:
                result_set = current_term_set
            else:
                result_set = result_set.intersection(current_term_set)
                
            if not result_set: # Early exit if intersection is empty
                return []
            
        return list(result_set) if result_set else []
