"""
Abstract base class for indexing modules.
"""

from abc import ABC, abstractmethod


class Index(ABC):
    """
    Abstract Base Class for an Index.
    """

    @abstractmethod
    def build_index(self, documents: dict[str, list[str]]) -> None:
        """
        Builds the index given a collection of parsed documents.

        Args:
            documents (dict[str, list[str]]): A mapping from document ID (or name)
                to a list of processed terms.
        """
        pass

    @abstractmethod
    def search(self, query_terms: list[str]) -> list[str]:
        """
        Searches the index for the given query terms.

        Args:
            query_terms (list[str]): A list of preprocessed terms from the query.

        Returns:
            list[str]: A list of document IDs that match the query.
        """
        pass
