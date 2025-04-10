from transformers import AutoTokenizer
from datasets import load_dataset

from tktkt.evaluation.context import *
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
from tktkt.util.iterables import take
from tktkt.util.types import NamedIterable


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
        "vmentob": 1
    }

    for word in word_corpus:
        print(word_tokeniser.prepareAndTokenise(word))

    accessors = getAccessors(word_tokeniser, word_corpus, 10000000000)
    for i in accessors.vocab.values():
        print(accessors.left_of.accessors[i])
        print(accessors.right_of.accessors[i])
        # print(k, "\t\t\t", av)


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
    from tktkt.util.printing import percent, dprint
    from tktkt.factories.preprocessing import KudoSpaceMarker, RobertaSpaceMarker
    def is_fullword(accessor: str, accessors: AccessorDistributions) -> bool:
        """You are a full word when you are known, but never seen with neighbours."""
        id = accessors.vocab[accessor]
        return (id not in accessors.left_of.accessors  or len(accessors.left_of.accessors[id]) == 0) \
           and (id not in accessors.right_of.accessors or len(accessors.right_of.accessors[id]) == 0)

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

    print("Loading corpus...")
    d = getCorpus(2000)
    tk = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("roberta-base"), for_single_words=True)
    accessors = getAccessors(tokeniser=tk, texts=d, bucket_samples_every=1000)

    # Analysis before filtering
    print("Pre-analysis...")
    summaries = analyseAccessors(accessors, do_count_ends_as_variety=True)
    print("> Accessors on the right, unweighted:")
    dprint(summaries.right.averages.__dict__, indent=1)
    print("> Accessors on the right, weighted:")
    dprint(summaries.right.weighted_averages.__dict__, indent=1)

    # Filters: first all weird tokens to get a distribution over just language.
    nonlanguage = [t for t in accessors.vocab if is_notlanguage(t, accessors)]
    n_nonlanguage = len(nonlanguage)
    print(f"Filtering {n_nonlanguage} types ({percent(n_nonlanguage, len(accessors.vocab))} of vocab) since they're not language (e.g. {','.join(nonlanguage[:5])})...")
    filterAccessors(accessors, is_notlanguage)

    # Compute the amount of full words that will be filtered in the next step, so we can display it as extra stat.
    fullwords = [t for t in accessors.vocab if is_fullword(t, accessors)]
    n_fullwords = len(fullwords)
    print(f"Filtering {n_fullwords} types ({percent(n_fullwords, len(accessors.vocab))} of remaining vocab) since they're full words (e.g. {','.join(fullwords[:5])})...")
    filterAccessors(accessors, is_fullword)

    # Analyse after filtering
    print("Analysis without full words and weird types...")
    summaries = analyseAccessors(accessors, do_count_ends_as_variety=True)
    print("> Accessors on the right, unweighted:")
    dprint(summaries.right.averages.__dict__, indent=1)
    print("> Accessors on the right, weighted:")
    dprint(summaries.right.weighted_averages.__dict__, indent=1)


if __name__ == "__main__":
    filtering()
