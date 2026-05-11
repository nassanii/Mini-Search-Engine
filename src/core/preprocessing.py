"""
Text preprocessing module.

This module is responsible for cleaning and preparing text data for indexing.
It applies operations such as case folding, tokenization, stop-word removal,
and lemmatization.
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded.
# In a production environment, this might be handled outside the application startup.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """
    A class that handles the preprocessing of text documents.
    """

    def __init__(self, language: str = 'english', method: str = 'lemmatization') -> None:
        """
        Initializes the TextPreprocessor.

        Args:
            language (str): The language for stop-words. Defaults to 'english'.
            method (str): Text normalization method: 'lemmatization' or 'stemming'.
        """
        self.stop_words: set[str] = set(stopwords.words(language))
        self.method = method
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stemmer: PorterStemmer = PorterStemmer()

    def process(self, text: str) -> list[str]:
        """
        Processes the input text through a pipeline of cleaning operations.

        Args:
            text (str): The raw text to process.

        Returns:
            list[str]: A list of cleaned and lemmatized tokens.
        """
        tokens = self._tokenize_and_casefold(text)
        tokens = self._remove_punctuation_and_stopwords(tokens)
        if self.method == 'stemming':
            normalized_tokens = self._stem(tokens)
        else:
            normalized_tokens = self._lemmatize(tokens)
        return normalized_tokens

    def _tokenize_and_casefold(self, text: str) -> list[str]:
        """
        Converts text to lowercase and tokenizes it.

        Args:
            text (str): The input text.

        Returns:
            list[str]: A list of lowercase tokens.
        """
        text = text.lower()
        return word_tokenize(text)

    def _remove_punctuation_and_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Removes punctuation tokens and stop-words from the list of tokens.

        Args:
            tokens (list[str]): The input tokens.

        Returns:
            list[str]: Filtered tokens.
        """
        filtered_tokens: list[str] = []
        for token in tokens:
            if token not in string.punctuation and token not in self.stop_words:
                filtered_tokens.append(token)
        return filtered_tokens

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Lemmatizes the given tokens.

        Args:
            tokens (list[str]): The input tokens.

        Returns:
            list[str]: Lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _stem(self, tokens: list[str]) -> list[str]:
        """
        Stems the given tokens.

        Args:
            tokens (list[str]): The input tokens.

        Returns:
            list[str]: Stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]
