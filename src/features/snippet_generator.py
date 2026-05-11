"""
Document snippet generator module.
"""

import os
from src.core.preprocessing import TextPreprocessor

class SnippetGenerator:
    """
    A class that generates snippets of text from documents surrounding query terms.
    """

    def __init__(self, documents_dir: str = "data/documents_split") -> None:
        """
        Initializes the SnippetGenerator.

        Args:
            documents_dir (str): The directory containing the text documents.
        """
        self.documents_dir = documents_dir
        self.preprocessor = TextPreprocessor(method='lemmatization')

    def get_snippet(self, doc_id: str, query_terms: list[str], window_size: int = 15) -> str:
        """
        Generates a text snippet for a given document around the query terms.

        Args:
            doc_id (str): The document ID (filename).
            query_terms (list[str]): The tokenized query terms.
            window_size (int): Number of words to include before and after the matched term.

        Returns:
            str: The generated snippet.
        """
        filepath = os.path.join(self.documents_dir, doc_id)
        if not os.path.exists(filepath):
            return ""

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            return ""

        words = text.split()
        lower_words = [w.lower().strip('.,!?;:()[]"') for w in words]
        
        query_set = set(self.preprocessor.process(" ".join(query_terms)))
        
        best_match_idx = -1
        max_matches_in_window = 0
        
        for i in range(len(words)):
            if lower_words[i] in query_set:
                start_idx = max(0, i - window_size)
                end_idx = min(len(words), i + window_size + 1)
                
                window_words = lower_words[start_idx:end_idx]
                matches = sum(1 for w in window_words if w in query_set)
                
                if matches > max_matches_in_window:
                    max_matches_in_window = matches
                    best_match_idx = i

        if best_match_idx == -1:
            return " ".join(words[:window_size * 2]) + "..."
            
        start_idx = max(0, best_match_idx - window_size)
        end_idx = min(len(words), best_match_idx + window_size + 1)
        
        snippet = " ".join(words[start_idx:end_idx])
        if start_idx > 0:
            snippet = "..." + snippet
        if end_idx < len(words):
            snippet = snippet + "..."
            
        return snippet
