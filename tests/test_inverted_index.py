"""
Tests for the indexing module.
"""

import unittest
from src.core.inverted_index import InvertedIndex

class TestInvertedIndex(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the test environment."""
        self.index = InvertedIndex()
        self.docs: dict[str, list[str]] = {
            "doc1": ["apple", "banana", "cherry"],
            "doc2": ["banana", "date", "fig"],
            "doc3": ["apple", "fig", "grape"]
        }

    def test_build_index(self) -> None:
        """Test building the inverted index."""
        self.index.build_index(self.docs)
        self.assertIn("apple", self.index.index)
        self.assertSetEqual(self.index.index["apple"], {"doc1", "doc3"})

    def test_search_single_term(self) -> None:
        """Test searching for a single term."""
        self.index.build_index(self.docs)
        results = self.index.search(["banana"])
        self.assertCountEqual(results, ["doc1", "doc2"])

    def test_search_multiple_terms(self) -> None:
        """Test searching with boolean AND logic."""
        self.index.build_index(self.docs)
        results = self.index.search(["apple", "banana"])
        self.assertEqual(results, ["doc1"])

if __name__ == '__main__':
    unittest.main()
