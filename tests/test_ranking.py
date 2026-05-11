"""
Tests for the search engine module.
"""

import unittest
from src.core.ranking import Ranker

class TestRanker(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the test environment."""
        self.ranker = Ranker()

    def test_get_cosine_similarity(self) -> None:
        """Test cosine similarity computation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        score = self.ranker.get_cosine_similarity(v1, v2)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
