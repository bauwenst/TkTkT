from tst.preamble import *

from transformers import AutoTokenizer
from datasets import load_dataset

from tktkt.evaluation.context import *
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
from tktkt.util.iterables import take
from tktkt.util.dicts import invertdict
from tktkt.util.types import NamedIterable
from tktkt.util.printing import percent, dprint
from tktkt.factories.preprocessing import KudoSpaceMarker, RobertaSpaceMarker


def smallCorpus():
    word_tokeniser = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("roberta-base"), for_single_words=True)
    word_corpus = {
        "discombobulated": 1,
        "disco": 1,
        "discontentment": 10,
        "dismemberment": 1,
        "displacement": 1,
        "qobment": 1,
        "zobment": 1,
        "zmentob": 1,
        "vmentob": 1,
        "x,y": 1  # This one is just here to check what happens in the CSV when the separator is a type.
    }

    print("Tokenised corpus:")
    for word in word_corpus:
        print(word_tokeniser.prepareAndTokenise(word))

    accessors = getAccessors(word_tokeniser, NamedIterable(list(word_corpus.items()), name="test"), 10000000000)
    for t,i in accessors.vocab.items():
        print(t)
        print(" Left:", list(accessors.left_of.accessors[i].values()), "+", accessors.left_of.boundaries[i])
        print("Right:", list(accessors.right_of.accessors[i].values()), "+", accessors.right_of.boundaries[i])
        print()

    s = analyseAccessors(accessors)
    print("LEFT")
    for t,d in s.left.per_type.items():
        print(t)
        print(d)

    print("\nRIGHT")
    for t,d in s.right.per_type.items():
        print(t)
        print(d)

    s.save()


def getCorpus(n: int):
    return NamedIterable(load_dataset("allenai/c4", "en", streaming=True)["train"], "test") \
         .map(lambda example: example["text"]) \
         .wrap(lambda iterator: take(n, iterator)) \
         .tqdm()


def assertLeftRightEquivalence():
    """
    Assert that for every token, there are an equal amount of (non-unique) left and right accessors.
    """
    d = getCorpus(2000)
    tk = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("roberta-base"), for_single_words=True)

    accessors = getAccessors(tk, d, 10000000000)
    for i in accessors.vocab.values():
        assert accessors.left_of.accessors[i].total() == accessors.right_of.accessors[i].total()


def summaries():
    """
    Print the summaries generated over a small corpus.
    """
    d = getCorpus(1000)
    tk = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("roberta-base"), for_single_words=True)
    accessors = getAccessors(tk, d, 20)
    results = analyseAccessors(accessors, do_count_ends_as_variety=False)
    for t,r in sorted(results.right.per_type.items(), key=lambda t: (t[1].av, t[0])):
        print(t)
        print("\t", r)
    print("Averages:", results.right.averages)
    print("Weighted averages:", results.right.weighted_averages)


def filtering():
    def is_fullword(accessor: str, accessors: AccessorDistributions) -> bool:
        """You are a full word when you are known, but never seen with neighbours."""
        id = accessors.vocab[accessor]
        return (id not in accessors.left_of.accessors  or len(accessors.left_of.accessors[id]) == 0) \
           and (id not in accessors.right_of.accessors or len(accessors.right_of.accessors[id]) == 0)

    def is_maybe_fullword(accessor: str, accessors: AccessorDistributions) -> bool:
        id = accessors.vocab[accessor]
        if id in accessors.left_of.accessors and len(accessors.left_of.accessors[id]) != 0 and accessors.left_of.boundaries[id] != 0 and \
                accessors.left_of.boundaries[id] / (accessors.left_of.accessors[id].total() + accessors.left_of.boundaries[id]) > 0.95:
            return True
        if id in accessors.right_of.accessors and len(accessors.right_of.accessors[id]) != 0 and accessors.right_of.boundaries[id] != 0 and \
                accessors.right_of.boundaries[id] / (accessors.right_of.accessors[id].total() + accessors.right_of.boundaries[id]) > 0.95:
            return True
        return False

    import regex
    pattern = regex.compile(r"\p{Punct}|\p{Digit}")
    def is_notlanguage(accessor: str, accessors: AccessorDistributions) -> bool:
        """
        You are NOT language when, absent any control characters, you contain punctuation or digits.
        """
        # TODO: When you use a non-Latinised byte encoding, what will happen here is that some subwords get flagged as
        #       bad (containing non-letters) because the original vocabularies used e.g. << for some letters.
        accessor_without_desirable_characters = accessor\
            .replace(RobertaSpaceMarker.substitute, "")\
            .replace(KudoSpaceMarker.substitute, "")\
            .replace(" ", "")
        return bool(pattern.search(accessor_without_desirable_characters))

    from tktkt.factories.preprocessing import TraditionalPreprocessor
    print("Loading corpus...")
    d = getCorpus(5000)
    tk = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("roberta-base"), for_single_words=True)
    highlights = {"Ġvelocity", "Ġretail", "Ġraw"}
    accessors = getAccessors(tokeniser=tk, texts=d, bucket_samples_every=1000, split_into_disjunct_examples=TraditionalPreprocessor(),
                             print_contexts_for_tokens=highlights)

    # Analysis before filtering
    print("Pre-analysis...")
    summaries = analyseAccessors(accessors, do_count_ends_as_variety=True)
    print("> Accessors on the right, unweighted:")
    dprint(summaries.right.averages.__dict__, indent=1)
    print("> Accessors on the right, weighted:")
    dprint(summaries.right.weighted_averages.__dict__, indent=1)

    # Filters: first all weird tokens to get a distribution over just language.
    print("Filtering non-language...")
    nonlanguage = accessors.filter(is_notlanguage)
    n_nonlanguage = len(nonlanguage)
    print(f"Filtered {n_nonlanguage} types ({percent(n_nonlanguage, len(accessors.vocab))} of vocab).")
    print(f"\tE.g.: {', '.join(nonlanguage)}")

    ###

    # The goal: find types which almost exclusively neighbour word boundaries.
    scores = dict()
    for t,i in accessors.vocab.items():
        a1 = accessors.left_of.accessors  .get(i, Counter()).total()
        b1 = accessors.left_of.boundaries .get(i, 0)
        a2 = accessors.right_of.accessors .get(i, Counter()).total()
        b2 = accessors.right_of.boundaries.get(i, 0)

        # Note: after filtering, there is no more guarantee that a1+b1 == a2+b2.
        scores[t] = min( b1 / (a1 + b1) if a1+b1 else float("inf"), b2 / (a2 + b2) if a2+b2 else float("inf") )

    for t in sorted(scores, key=scores.get):
        print(f"{t} ({accessors.vocab[t]}) | {scores[t]}")

    ivocab = invertdict(accessors.vocab)
    for t in highlights:
        print(t)
        print("\t", {ivocab[k]: v for k,v in  accessors.left_of.accessors[accessors.vocab[t]].items()})
        print("\t", {ivocab[k]: v for k,v in accessors.right_of.accessors[accessors.vocab[t]].items()})

    # print([t for t in accessors.vocab if is_fullword(t,accessors)])
    # print([t for t in accessors.vocab if is_maybe_fullword(t,accessors)])
    # quit()
    ###

    # Now all the full words.
    #   - Note: it is impossible that a type qualifies as a full word after this step, because that means it had at
    #     least one filtered type next to it, which means that filtered type shouldn't have been filtered.
    fullwords = accessors.filter(is_fullword)
    n_fullwords = len(fullwords)
    print(f"Filtering {n_fullwords} types ({percent(n_fullwords, len(accessors.vocab))} of remaining vocab).")
    print(f"\tE.g.: {', '.join(fullwords)}")

    # Analyse after filtering
    print("Analysis without full words and weird types...")
    summaries = analyseAccessors(accessors, do_count_ends_as_variety=True)
    print("> Accessors on the right, unweighted:")
    dprint(summaries.right.averages.__dict__, indent=1)
    print("> Accessors on the right, weighted:")
    dprint(summaries.right.weighted_averages.__dict__, indent=1)


if __name__ == "__main__":
    smallCorpus()
