from tktkt.util.trie import *
from tktkt.util.printing import lprint


def prefix():
    trie = PrefixTrieNode()
    trie.add("abc", 3)
    trie.add("ab", 1)
    trie.add("ac", 2)
    trie.add("abd", 1)
    trie.add("abcd", 1)
    trie.add("b", 1)

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


def suffix():
    print("=" * 50)

    trie = SuffixTrieNode()
    trie.add("abc", 3)
    trie.add("ab", 1)
    trie.add("ac", 2)
    trie.add("abd", 1)
    trie.add("abcd", 1)
    trie.add("b", 1)

    print(trie)
    print(trie.get("c"))
    # print(trie.getNodesOfSuffices("abca"))

    print("---")
    lprint(trie.getNodesOfSuffices("bc"))
    print("---")
    lprint(trie.getNodesOfSuffices("cb"))
    print("---")
    print(trie.getNodesOfSuffices("a"))
    print("---")

    trie.compile()
    print(trie)
