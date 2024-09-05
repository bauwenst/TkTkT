from tqdm.auto import tqdm

from tktkt.util.combinatorics import *
from tktkt.util.printing import lprint


def test_computePermutationNumber():
    segmentation = (1, 1, 1, 2, 2, 4)
    lprint(sorted(map(lambda p: (permutationToIdentifier(p), p), permutations_no_repeat(segmentation))))


def test_globalIdentifiers():
    for word_length in tqdm(range(20)):
        total = int(2**(word_length-1))
        target_identifiers = set(range(total))
        unique_identifiers = {getLOCKey(t) for t in tqdm(compositions(word_length), total=total)}
        assert target_identifiers == unique_identifiers


if __name__ == "__main__":
    test_globalIdentifiers()
