from transformers import AutoTokenizer
from datasets import load_dataset

from tktkt.evaluation.context import getAccessors, analyseAccessors
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



if __name__ == "__main__":
    summaries()