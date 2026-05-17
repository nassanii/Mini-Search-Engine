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
        # Start with the original terms in a set to avoid duplicates
        expanded_terms = set(query_terms)
        
        # Iterate over each word in the query
        for term in query_terms:
            # Get all synsets (groups of synonymous words) from WordNet for this term
            for syn in wordnet.synsets(term):
                # Extract individual words (lemmas) from each synset
                for lemma in syn.lemmas():
                    # Add the synonymous word in lowercase to our expanded set
                    expanded_terms.add(lemma.name().lower())
                    
        return list(expanded_terms)
