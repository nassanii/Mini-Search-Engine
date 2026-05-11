"""
Spell checker module.
"""

from spellchecker import SpellChecker as PySpellChecker

class SpellChecker:
    """
    A class that handles spelling correction using pyspellchecker.
    It builds a custom dictionary from the corpus vocabulary.
    """

    def __init__(self) -> None:
        """Initializes the SpellChecker."""
        self.checker = PySpellChecker(distance=2)

    def load_vocabulary(self, vocabulary: list[str]) -> None:
        """
        Adds the corpus vocabulary to the spell checker's dictionary.

        Args:
            vocabulary (list[str]): List of unique words in the corpus.
        """
        self.checker.word_frequency.load_words(vocabulary)

    def get_corrections(self, query_terms: list[str]) -> dict[str, str]:
        """
        Checks for misspelled words in the query and returns corrections.

        Args:
            query_terms (list[str]): The tokenized query.

        Returns:
            dict[str, str]: A dictionary mapping misspelled words to their corrections.
        """
        misspelled = self.checker.unknown(query_terms)
        corrections = {}
        for word in misspelled:
            correction = self.checker.correction(word)
            if correction and correction != word:
                corrections[word] = correction
        return corrections
