from typing import Dict, List, Optional


class TrieNode:
    """
    Simple character-based trie (uncompiled) or word trie (compiled).
    """

    def __init__(self, root: str=""):
        self._compiled = False
        self._compiledroots = False

        self.root = root
        self.count = 0
        self.branches: Dict[str, TrieNode] = dict()

        self.lexicographic_branch_keys: List[str] = []

    def add(self, word: str, count: int=1):
        """
        Add a count to the current node and to a word path starting UNDER this node.
        """
        if self._compiled:
            raise RuntimeError("Tried to a alter a compiled trie.")

        self.count += count
        if not word:
            return

        next_character = word[0]
        if next_character not in self.branches:
            self.branches[next_character] = TrieNode(next_character)  # Bidirectional association

        self.branches[next_character].add(word[1:], count)

    def compile(self):
        """
        Reduces the trie such that every node corresponds to a word that was once added.

        Finds children whose own children make up 100% of their size, which means that the child is useless and
        that the current Trie should own its grandchildren as children.
        Compilation happens from deep to shallow nodes so that we know the inherited grandchildren aren't themselves redundant.
        """
        if self._compiledroots:
            raise RuntimeError("Cannot compile trie after compiling its roots. Reverse the order of these two operations.")

        if self._compiled:
            return

        for child_root, child in dict(self.branches).items():
            child.compile()
            assert child_root == child.root

            if child.count == sum(grandchild.count for grandchild in child.branches.values()):
                self.branches.pop(child.root)
                for grandchild in child.branches.values():
                    grandchild.root = child.root + grandchild.root
                    self.branches[grandchild.root] = grandchild

        self.lexicographic_branch_keys = sorted(self.branches, key=lambda key: (len(key), key))

        self._compiled = True
        self._assertBidirectionalAssociation()

    def compileRoots(self):
        """
        By default, the trie stores string segments in its nodes such that the string that leads to a node is only
        obtained by SUMMING across all string segments on the path from the root to that node.

        This transformation instead stores those full strings in each node.
        """
        for child in self.branches.values():
            child.root = self.root + child.root
            child.compileRoots()
        self._compiledroots = True

    def _assertBidirectionalAssociation(self):
        for stored_root, child in self.branches.items():
            assert stored_root == child.root

    def __repr__(self):
        result = f"{self.root} {self.count}\n"
        for char in sorted(self.branches.keys()):
            result += prefixLines(self.branches[char].__repr__(), "|\t")
        return result

    def get(self, prefix: str) -> Optional["TrieNode"]:
        """
        Get the descendant node you would get by walking down the children of this node recursively matching the given string.
        """
        if not prefix:
            return self

        for key in self.branches:
            if prefix.startswith(key):  # Can only be true for exactly one branch, because if the given word startswith two different keys, one of those keys is a prefix for the other, and would be a child of that one.
                return self.branches[key].get(prefix[len(key):])

        return None

    def getNodesOfPrefices(self, word: str) -> List["TrieNode"]:
        """
        Get the list of existing trie nodes that correspond to a prefix of the given word.
        For example, "abcde" can return the nodes for "a", "ab", "abc", "abcd" and "abcde".
        """
        nodes = []
        for i in range(len(word)):
            node = self.get(word[:i+1])
            if node is not None:
                nodes.append(node)
        return nodes

    def getNodesWithPrefix(self, word: str, only_first: bool=False) -> List["TrieNode"]:
        """
        Return the nodes that start with the given string.
        For example, if "abc" is given, the node for "abcd" and "abce" might be returned, even if "abc" itself doesn't
        have its own node.
        """
        nodes = []
        if not word:  # The word is exactly in the trie.
            nodes.append(self)
            if not only_first:
                nodes.extend(self.getDescendantNodes())
            return nodes

        for key in self.lexicographic_branch_keys:
            node = self.branches[key]
            if len(key) <= len(word):  # Then you have to match all the characters in the key and continue going deeper.
                if word.startswith(key):  # Can only happen once, see above.
                    return node.getNodesWithPrefix(word[len(key):], only_first=only_first)
            else:
                if key.startswith(word):  # Not possible when there is another key the word starts with, because then that key would also be a prefix for the current key.
                    nodes.append(node)
                    if only_first:
                        return nodes
                    nodes.extend(node.getDescendantNodes())

        return nodes

    def getDescendantNodes(self) -> List["TrieNode"]:
        descendants = []
        for node in self.branches.values():
            descendants.append(node)
            descendants.extend(node.getDescendantNodes())
        return descendants


def prefixLines(s: str, prefix: str="\t") -> str:
    return "".join([prefix + line + "\n" for line in s.splitlines()])


if __name__ == "__main__":
    trie = TrieNode()
    trie.add("abc", 3)
    trie.add("ab",  1)
    trie.add("ac",  2)
    trie.add("abd", 1)
    trie.add("abcd", 1)
    trie.add("b",   1)

    print(trie)
    print(trie.get("ab"))

    trie.compile()
    trie._assertBidirectionalAssociation()
    print(trie)
    print(trie.get("a"))

    print([node.root for node in trie.getNodesOfPrefices("abc")])
    trie.compileRoots()
    print([node.root for node in trie.getNodesOfPrefices("abcefgh")])
    print([node.root for node in trie.getNodesWithPrefix("a", only_first=True)])
