"""Union-Find (Disjoint Set Union) data structure for efficient net merging.

This provides O(α(n)) amortized time complexity for union and find operations,
where α is the inverse Ackermann function (effectively constant for practical inputs).

This is a significant improvement over the O(n) net merging in the original
Circuit.connect() implementation.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class UnionFind(Generic[T]):
    """Disjoint Set Union with path compression and union by rank.

    Optimized for net merging in circuit connectivity tracking.
    """

    __slots__ = ("_parent", "_rank", "_canonical")

    def __init__(self) -> None:
        # Maps element -> parent element
        self._parent: dict[T, T] = {}
        # Maps root element -> tree rank (for union by rank)
        self._rank: dict[T, int] = {}
        # Maps root -> canonical value (for preserving named nets)
        self._canonical: dict[T, T] = {}

    def make_set(self, x: T, canonical: T | None = None) -> None:
        """Create a new singleton set containing x.

        Args:
            x: Element to add
            canonical: Optional canonical value to preserve (e.g., named Net)
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._canonical[x] = canonical if canonical is not None else x

    def find(self, x: T) -> T:
        """Find the representative of x's set with path compression.

        Returns:
            The root/representative of x's set

        Raises:
            KeyError: If x is not in any set
        """
        if x not in self._parent:
            raise KeyError(f"Element {x!r} not in any set")

        # Path compression: make every node on path point directly to root
        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        # Compress path
        current = x
        while current != root:
            next_parent = self._parent[current]
            self._parent[current] = root
            current = next_parent

        return root

    def union(self, x: T, y: T, prefer: T | None = None) -> T:
        """Merge the sets containing x and y.

        Args:
            x: First element
            y: Second element
            prefer: If provided, prefer this element's canonical value

        Returns:
            The new root element
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return root_x

        # Union by rank: attach smaller tree under larger
        rank_x = self._rank[root_x]
        rank_y = self._rank[root_y]

        # Determine which root to keep based on preference and rank
        if prefer is not None:
            prefer_root = self.find(prefer)
            if prefer_root == root_x:
                # Keep root_x as the new root
                self._parent[root_y] = root_x
                if rank_x == rank_y:
                    self._rank[root_x] += 1
                return root_x
            elif prefer_root == root_y:
                # Keep root_y as the new root
                self._parent[root_x] = root_y
                if rank_x == rank_y:
                    self._rank[root_y] += 1
                return root_y

        # Default union by rank
        if rank_x < rank_y:
            self._parent[root_x] = root_y
            # Preserve canonical from root_x if it's a named net
            canon_x = self._canonical.get(root_x)
            canon_y = self._canonical.get(root_y)
            if canon_x is not None and canon_y is None:
                self._canonical[root_y] = canon_x
            return root_y
        elif rank_x > rank_y:
            self._parent[root_y] = root_x
            canon_x = self._canonical.get(root_x)
            canon_y = self._canonical.get(root_y)
            if canon_y is not None and canon_x is None:
                self._canonical[root_x] = canon_y
            return root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1
            canon_x = self._canonical.get(root_x)
            canon_y = self._canonical.get(root_y)
            if canon_y is not None and canon_x is None:
                self._canonical[root_x] = canon_y
            return root_x

    def get_canonical(self, x: T) -> T:
        """Get the canonical value for x's set.

        Returns:
            The canonical value (e.g., named Net) for the set containing x
        """
        root = self.find(x)
        return self._canonical.get(root, root)

    def set_canonical(self, x: T, canonical: T) -> None:
        """Set the canonical value for x's set."""
        root = self.find(x)
        self._canonical[root] = canonical

    def connected(self, x: T, y: T) -> bool:
        """Check if x and y are in the same set."""
        try:
            return self.find(x) == self.find(y)
        except KeyError:
            return False

    def __contains__(self, x: T) -> bool:
        return x in self._parent

    def __len__(self) -> int:
        """Return number of elements (not number of sets)."""
        return len(self._parent)
