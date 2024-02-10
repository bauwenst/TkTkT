"""
TODO: Two operations to support. I know the string that follows at the point I'm at. Now,
    x For concatenable Viterbi: for each prefix in that string, I need to get it if it exists in the vocabulary.
    - For non-concatenable Viterbi: for each prefix in that string, I need to get all subwords that start with that prefix.
"""
from typing import Dict


class TrieNode:
    """
    Simple character-based trie (uncompiled) or word trie (compiled).
    """

    def __init__(self, root: str=""):
        self.compiled = False
        self.compiledroots = False

        self.root = root
        self.count = 0
        self.branches: Dict[str, TrieNode] = dict()

    def add(self, word: str, count: int=1):
        """
        Add a count to the current node and to a word path starting UNDER this node.
        """
        if self.compiled:
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
        if self.compiledroots:
            raise RuntimeError("Cannot compile trie after compiling its roots. Reverse the order of these two operations.")

        if self.compiled:
            return

        for child_root, child in dict(self.branches).items():
            child.compile()
            assert child_root == child.root

            if child.count == sum(grandchild.count for grandchild in child.branches.values()):
                self.branches.pop(child.root)
                for grandchild in child.branches.values():
                    grandchild.root = child.root + grandchild.root
                    self.branches[grandchild.root] = grandchild

        self.compiled = True
        self._assertBidirectionalAssociation()

    def _assertBidirectionalAssociation(self):
        for stored_root, child in self.branches.items():
            assert stored_root == child.root

    def __repr__(self):
        result = f"{self.root} {self.count}\n"
        for char in sorted(self.branches.keys()):
            result += prefixLines(self.branches[char].__repr__(), "|\t")
        return result

    def get(self, prefix: str) -> "TrieNode":
        """
        Get the node corresponding to the given child path.
        """
        if not prefix:
            return self

        for key in self.branches:
            if prefix.startswith(key):
                return self.branches[key].get(prefix[len(key):])

        return None

    def getNodesOfPrefices(self, word: str):
        nodes = []
        for i in range(len(word)):
            node = self.get(word[:i+1])
            if node is not None:
                nodes.append(node)
        return nodes

    def compileRoots(self):
        for child in self.branches.values():
            child.root = self.root + child.root
            child.compileRoots()
        self.compiledroots = True


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
    print([node.root for node in trie.getNodesOfPrefices("abc")])
