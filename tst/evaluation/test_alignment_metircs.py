from tktkt.evaluation.morphological import *


def test_dilution():
    import re

    p = re.compile("A+|B+|C+|D+|E+|F+|G+|H+")

    examples = [
        "AAABBBCCC",
        "AAABBB|CCC",
        "AAA|BBBCCC",
        "AAA|BBB|CCC",
        "A|AA|BBB",
        "AAAB|BB",
        "A|AAB|BB",
        "A|A|ABBB",
        "A|AAABBB",
        "AAABBBC|CC",
        "A|AABBBC|CC",
        "A|AABBB|CCC",
        "A|AABBB|CCC",
        "A|AABBBCCC|DDD",
        "A|AABBBCC|CDDD",
        "AA|ABBBCC|CDDDEEEF|FF",
        "AAA|BBBCCC|DDDEEEFFF",
        "AA|AAA|AA|BBBC|CCDDD|EF|FFGGGHH"
    ]

    for example in examples:
        tokens = example.split("|")
        morphs = p.findall(example.replace("|", ""))

        print(example)
        turbid = NestedAverage()
        muddy = NestedAverage()
        dilute = NestedAverage()
        morphDilution(morphs, tokens, turbid, dilute, muddy)  # Uncomment prints here to see interesting stuff.
        print(turbid.compute())
        print(dilute.compute())
        print(muddy.compute())
        print()


if __name__ == "__main__":
    test_dilution()
