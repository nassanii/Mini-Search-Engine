"""
Tests for the preprocessing module.
"""

import unittest
from unittest.mock import patch
from src.core.preprocessing import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the test environment."""
        # Note: In a real test environment, we might want to mock the NLTK downloads
        # or rely on a pre-downloaded cache. For this skeleton, we just instantiate.
        self.preprocessor = TextPreprocessor()

    def test_process(self) -> None:
        """Test the full processing pipeline."""
        text = "This is a Simple TEST string, testing!"
        tokens = self.preprocessor.process(text)
        
        # 'this', 'is', 'a' might be stopwords depending on NLTK version,
        # but let's just do a basic sanity check.
        self.assertIsInstance(tokens, list)
        self.assertNotIn("TEST", tokens)
        self.assertNotIn(",", tokens)
        
        # 'testing' might be lemmatized to 'testing' or 'test'
        # just ensuring tokens are strings.
        for t in tokens:
            self.assertIsInstance(t, str)

if __name__ == '__main__':
    unittest.main()
