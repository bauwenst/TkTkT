import re

from tktkt.util.timing import timeit
from tktkt.evaluation.morphological import compareSplits_cursors

from bpe_knockout.project.config import morphologyGenerator


def compareSplits_tokenLengths(candidate: str, reference: str):
    # Numpy implementation is short, but an order of magnitude slower!
    # c_indices = np.cumsum([len(t) for t in candidate.split()]) - 1
    # r_indices = np.cumsum([len(t) for t in reference.split()]) - 1
    c_indices = [len(t) for t in candidate.split()]
    r_indices = [len(t) for t in reference.split()]
    cum = 0
    for i in range(len(c_indices)):
        cum += c_indices[i]
        c_indices[i] = cum
    cum = 0
    for i in range(len(r_indices)):
        cum += r_indices[i]
        r_indices[i] = cum

    tp = len(set(c_indices) & set(r_indices)) - 1
    relevant  = len(r_indices) - 1
    predicted = len(c_indices) - 1
    total = c_indices[-1] - 1
    return tp, predicted, relevant, total


SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))
def compareSplits_regex(candidate: str, reference: str):
    c = " ".join(candidate.strip()).replace("   ", SPLIT_MARKER)
    r = " ".join(reference.strip()).replace("   ", SPLIT_MARKER)

    c_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(c)}
    r_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(r)}

    tp = len(c_indices & r_indices)
    relevant = len(r_indices)
    predicted = len(c_indices)
    total = len(r) // 2
    return tp, predicted, relevant, total



def generateExamples(N, L, P):
    from string import ascii_lowercase
    import numpy.random as npr

    npr.seed(0)

    longest_word = ascii_lowercase*(L // 26) + ascii_lowercase[:L % 26]

    examples = []
    for i in range(N):
        l = npr.randint(1,len(longest_word))
        word = longest_word[:l]

        candidate_split = ""
        reference_split = ""
        for j in range(l):
            candidate_split += word[j]
            reference_split += word[j]

            if j != l-1:
                if npr.random() < P:
                    candidate_split += " "

                if npr.random() < P:
                    reference_split += " "

        examples.append((candidate_split, reference_split))

    print(examples)
    return examples



# EXAMPLES = [("ab c d efghi j", "a bcd ef ghij"),
#             ("abc def", "a bcd e f")]
EXAMPLES = generateExamples(20, 40, 0.75)


def correctness():
    for candidate, reference in EXAMPLES:
        print(compareSplits_cursors(candidate, reference))
        print(compareSplits_tokenLengths(candidate, reference))
        print(compareSplits_regex(candidate, reference))
        print()


@timeit
def speed(method):
    for candidate, reference in 250_000*EXAMPLES:
        method(candidate, reference)


if __name__ == "__main__":
    correctness()  # Also does numba compilation if you need it.

    print()
    speed(compareSplits_cursors)
    print("^cursors")

    speed(compareSplits_tokenLengths)
    print("^lengths")

    speed(compareSplits_regex)
    print("^regex")
