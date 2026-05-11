"""
Search engine ranking module.

This module provides functionalities for ranking search results,
including computing TF-IDF scores and Cosine Similarity between
documents and queries.
"""

import math
from collections import Counter
import numpy as np


class Ranker:
    """
    A class responsible for ranking documents based on TF-IDF and Cosine Similarity.
    """

    def __init__(self) -> None:
        """Initializes the Ranker."""
        self.tf_idf_matrix: dict[str, dict[str, float]] = {}
        self.idf: dict[str, float] = {}
        self.vocabulary: list[str] = []

    def compute_tf_idf(self, documents: dict[str, list[str]]) -> None:
        """
        Computes the TF-IDF matrix for a collection of documents.

        Args:
            documents (dict[str, list[str]]): A mapping from document IDs to lists of terms.
        """
        N = len(documents)
        df: dict[str, int] = {}
        tf_counts: dict[str, dict[str, int]] = {}
        vocab_set = set()
        
        # Calculate TF and DF
        for doc_id, terms in documents.items():
            counts = Counter(terms)
            tf_counts[doc_id] = dict(counts)
            for term in counts.keys():
                df[term] = df.get(term, 0) + 1
                vocab_set.add(term)
                
        self.vocabulary = list(vocab_set)
        
        # Calculate IDF
        for term, doc_freq in df.items():
            self.idf[term] = math.log(N / doc_freq)
            
        # Calculate TF-IDF
        for doc_id, counts in tf_counts.items():
            self.tf_idf_matrix[doc_id] = {}
            for term, count in counts.items():
                self.tf_idf_matrix[doc_id][term] = count * self.idf[term]

    def get_cosine_similarity(self, query_vector: list[float], doc_vector: list[float]) -> float:
        """
        Computes the cosine similarity between two vectors using numpy.

        Args:
            query_vector (list[float]): The TF-IDF vector of the query.
            doc_vector (list[float]): The TF-IDF vector of a document.

        Returns:
            float: The cosine similarity score.
        """
        q = np.array(query_vector)
        d = np.array(doc_vector)
        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)
        if norm_q == 0 or norm_d == 0:
            return 0.0
        return float(np.dot(q, d) / (norm_q * norm_d))

    def rank_documents(self, query_terms: list[str]) -> list[tuple[str, float]]:
        """
        Ranks documents for a given query based on cosine similarity.

        Args:
            query_terms (list[str]): The processed query terms.

        Returns:
            list[tuple[str, float]]: A list of tuples containing document IDs and their scores,
                sorted in descending order of score.
        """
        query_counts = Counter(query_terms)
        query_tf_idf = {}
        for term, count in query_counts.items():
            if term in self.idf:
                query_tf_idf[term] = count * self.idf[term]
                
        if not query_tf_idf:
            return []
            
        q_vec = [query_tf_idf.get(t, 0.0) for t in self.vocabulary]
        
        results: list[tuple[str, float]] = []
        for doc_id, doc_tf_idf in self.tf_idf_matrix.items():
            if any(term in doc_tf_idf for term in query_tf_idf):
                d_vec = [doc_tf_idf.get(t, 0.0) for t in self.vocabulary]
                score = self.get_cosine_similarity(q_vec, d_vec)
                if score > 0:
                    results.append((doc_id, score))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        return results
