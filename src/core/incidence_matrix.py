"""
Incidence Matrix implementation.
"""

import pandas as pd

class IncidenceMatrix:
    """
    Incidence Matrix implementation of an Index using Pandas.
    """

    def __init__(self) -> None:
        """Initializes an empty Incidence Matrix."""
        self.df: pd.DataFrame | None = None
        self.documents_ids: list[str] = []

    def build_index(self, documents: dict[str, list[str]]) -> None:
        """
        Builds the Incidence Matrix DataFrame.
        This runs after preprocessing and builds the full binary table.

        Args:
            documents (dict[str, list[str]]): Documents to index.
        """
        self.documents_ids = list(documents.keys())
        
        # Extract the complete unique vocabulary across all documents
        vocab = set()
        for terms in documents.values():
            vocab.update(terms)
            
        vocab_list = list(vocab)
        
        # Build binary mapping: 1 if term exists in a specific document, 0 otherwise
        matrix_data = {}
        for doc_id, terms in documents.items():
            doc_terms_set = set(terms)
            # Create a list of 1s and 0s for each word in the global vocabulary
            matrix_data[doc_id] = [1 if term in doc_terms_set else 0 for term in vocab_list]
            
        # Create Pandas DataFrame where Rows = Vocabulary Terms, and Columns = Document IDs
        self.df = pd.DataFrame(matrix_data, index=vocab_list)

    def get_term_vector(self, term: str) -> list[int]:
        """
        Retrieves the binary vector (row) for a specific term.
        
        Args:
            term (str): The search term.
            
        Returns:
            list[int]: A bit-vector representing document occurrences.
        """
        if self.df is None or term not in self.df.index:
            return [0] * len(self.documents_ids)
            
        return self.df.loc[term].tolist()

    def search(self, query_terms: list[str]) -> list[str]:
        """
        Searches using boolean AND on the Incidence Matrix.

        Args:
            query_terms (list[str]): Search terms.

        Returns:
            list[str]: Matching document IDs.
        """
        # Return empty if matrix isn't built or no query terms
        if self.df is None or not query_terms:
            return []
            
        # If any query term is missing from the entire vocabulary, there's no way it matches (AND logic)
        for term in query_terms:
            if term not in self.df.index:
                return []
                
        # Perform Bitwise AND across all row vectors corresponding to query terms
        # This yields a single boolean series indicating which documents contain ALL query terms
        result_vector = self.df.loc[query_terms].all(axis=0)
        
        # Extract and return the names of document IDs where the result is True
        matching_docs = result_vector[result_vector].index.tolist()
        return matching_docs
