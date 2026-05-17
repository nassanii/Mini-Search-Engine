"""
Inverted Index implementation.
"""
import fnmatch


class InvertedIndex:
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
        # Loop over every document and its terms
        for doc_id, terms in documents.items():
            for term in terms:
                # If the term is encountered for the first time, initialize an empty set for it
                if term not in self.index:
                    self.index[term] = set()
                # Add the document ID to the posting list (set) of this term
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
        # If no query terms are provided, return an empty list
        if not query_terms:
            return []
        
        result_set = None
        
        # Iterate through each term in the search query
        for term in query_terms:
            current_term_set = set()
            
            # Check if the term contains wildcard characters
            if '*' in term or '?' in term:
                # Find all vocabulary terms that match the wildcard pattern
                matches = self.get_wildcard_matches(term)
                # Combine the posting lists of all matched terms
                for match in matches:
                    current_term_set.update(self.index.get(match, set()))
            else:
                # Get the posting list for the exact term
                current_term_set = self.index.get(term, set())
                
            # If this is the first term, initialize the result set
            if result_set is None:
                result_set = current_term_set
            else:
                # Perform an intersection (AND logic) to keep only documents that contain ALL terms so far
                result_set = result_set.intersection(current_term_set)
                
            # Early exit: if the intersection becomes empty at any point, no documents match the full query
            if not result_set: 
                return []
            
        return list(result_set) if result_set else []
