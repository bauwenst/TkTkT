from transformers import AutoTokenizer

from tktkt.evaluation.context import getAccessorVarieties
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser


def dummy_test():
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

    for k, av in getAccessorVarieties(word_tokeniser, word_corpus, do_count_ends=False).items():
        print(k, "\t\t\t", av)


if __name__ == "__main__":
    dummy_test()