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
        # Step 1: Lowercase and split the text into words
        tokens = self._tokenize_and_casefold(text)
        
        # Step 2: Filter out punctuation marks and common stop-words
        tokens = self._remove_punctuation_and_stopwords(tokens)
        
        # Step 3: Normalize words using either stemming or lemmatization
        if self.method == 'stemming':
            normalized_tokens = self._stem(tokens)
        else:
            normalized_tokens = self._lemmatize(tokens)
            
        return normalized_tokens

    def _tokenize_and_casefold(self, text: str) -> list[str]:
        # Step 1: Convert text to lowercase and tokenize it
        print("[Preprocessing] Step 1: Tokenizing and Case Folding...")
        text = text.lower()
        return word_tokenize(text)

    def _remove_punctuation_and_stopwords(self, tokens: list[str]) -> list[str]:
        # Step 2: Remove punctuation and stop-words
        print("[Preprocessing] Step 2: Removing Punctuation and Stop-words...")
        filtered_tokens: list[str] = []
        # Iterate over all tokens and keep only valid words
        for token in tokens:
            # Check if the token is not a punctuation symbol AND not a stop-word
            if token not in string.punctuation and token not in self.stop_words:
                filtered_tokens.append(token)
        return filtered_tokens

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        # Step 3: Return words to their base form (Lemmatization)
        print("[Preprocessing] Step 3: Lemmatizing tokens...")
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _stem(self, tokens: list[str]) -> list[str]:
        # Step 3: Stem the words (Stemming)
        print("[Preprocessing] Step 3: Stemming tokens...")
        return [self.stemmer.stem(token) for token in tokens]
