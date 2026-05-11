"""
Query expansion module.
"""

from nltk.corpus import wordnet

class QueryExpander:
    """
    A class that expands a query using WordNet synonyms.
    """

    def expand_query(self, query_terms: list[str]) -> list[str]:
        """
        Expands the query terms by adding their synonyms.

        Args:
            query_terms (list[str]): The original tokenized query.

        Returns:
            list[str]: The expanded list of query terms including synonyms.
        """
        expanded_terms = set(query_terms)
        for term in query_terms:
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    expanded_terms.add(lemma.name().lower())
        return list(expanded_terms)
