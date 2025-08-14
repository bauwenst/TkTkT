from typing import Dict, List, Optional

from .strings import indent


class PrefixTrieNode:
    """
    Simple character prefix trie (uncompiled) or word prefix trie (compiled).
    """

    def __init__(self, root: str=""):
        self._compiled = False
        self._compiledroots = False

        self.root = root
        self.count = 0
        self.branches: Dict[str, PrefixTrieNode] = dict()

        self.lexicographic_branch_keys: List[str] = []  # Branch keys with a definite order independent of when which branch was added. The order is from short to long and then alphabetic.

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
            self.branches[next_character] = PrefixTrieNode(next_character)  # Bidirectional association

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

    def __repr__(self, sort_alphabetically: bool=False):
        result = f"{self.root} {self.count}\n"
        for char in sorted(self.branches.keys(), key=(lambda name: name) if sort_alphabetically else (lambda name: self.branches[name].count), reverse=not sort_alphabetically):
            result += indent(1, self.branches[char].__repr__(sort_alphabetically), tab="|\t")
        return result

    def get(self, prefix: str) -> Optional["PrefixTrieNode"]:
        """
        Get the descendant node you would get by walking down the children of this node recursively matching the given string.
        """
        if not prefix:
            return self

        for key in self.branches:
            if prefix.startswith(key):  # Can only be true for exactly one branch, because if the given word startswith two different keys, one of those keys is a prefix for the other, and would be a child of that one.
                return self.branches[key].get(prefix[len(key):])

        return None

    def getTopChildNodes(self, n: int=-1) -> List["PrefixTrieNode"]:
        nodes = sorted(self.branches.values(), key=lambda node: node.count, reverse=True)
        return nodes if n <= 0 else nodes[:n]

    def getNodesOfPrefices(self, word: str) -> List["PrefixTrieNode"]:
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

    def getNodesWithPrefix(self, word: str, only_first: bool=False) -> List["PrefixTrieNode"]:
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

    def getDescendantNodes(self) -> List["PrefixTrieNode"]:
        """
        Return the list of strict descendants of this node.
        """
        descendants = []
        for node in self.branches.values():
            descendants.append(node)
            descendants.extend(node.getDescendantNodes())
        return descendants


class SuffixTrieNode:
    """
    Suffix trie. The underlying data structure is still a prefix trie; this is just a wrapper that reverses inputs given
    to that trie. Whenever a method is supposed to return nodes, those nodes are wrapped on-the-fly to give the illusion
    that the entire trie consists of suffix nodes.
    """

    def __init__(self, backend: PrefixTrieNode=None):
        if backend is None:
            backend = PrefixTrieNode()
        self._node = backend

    @property
    def root(self):
        return self._node.root[::-1]

    @property
    def count(self):
        return self._node.count

    def compile(self):
        self._node.compile()

    def compileRoots(self):
        self._node.compileRoots()

    def add(self, word: str, count: int=1):
        return self._node.add(word=word[::-1], count=count)

    def get(self, suffix: str) -> Optional["SuffixTrieNode"]:
        node = self._node.get(prefix=suffix[::-1])
        if node is not None:
            node = SuffixTrieNode(node)
        return node

    def __repr__(self, sort_alphabetically: bool=False):
        return self._node.__repr__(sort_alphabetically)

    def getTopChildNodes(self, n: int=-1) -> List["SuffixTrieNode"]:
        return [SuffixTrieNode(node) for node in self._node.getTopChildNodes(n)]

    def getNodesOfSuffices(self, word: str) -> List["SuffixTrieNode"]:
        nodes = []
        for i in range(len(word)):
            node = self.get(word[len(word)-(i+1):])
            if node is not None:
                nodes.append(node)
        return nodes

    def getNodesWithSuffix(self, word: str, only_first: bool=False) -> List["SuffixTrieNode"]:
        """
        Return the nodes that end with the given string.
        For example, if "abc" is given, the node for "dabc" and "eabc" might be returned, even if "abc" itself doesn't
        have its own node.
        """
        nodes = self._node.getNodesWithPrefix(word[::-1], only_first=only_first)
        return [SuffixTrieNode(node) for node in nodes]

    def getDescendantNodes(self) -> List["SuffixTrieNode"]:
        nodes = self._node.getDescendantNodes()
        return [SuffixTrieNode(node) for node in nodes]


PrefixTrie = PrefixTrieNode
SuffixTrie = SuffixTrieNode
