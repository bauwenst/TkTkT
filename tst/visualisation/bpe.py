
def minimalWorkingExample():
    """
    Load a RoBERTa-base BPE tokeniser and visualise its BPE trees.
    """
    from tktkt.visualisation.bpe.trees import BpeVisualiser

    from tktkt.factories.tokenisers import Factory_BPE_Pythonic
    from tktkt.factories.deserialisation import BPE50k_RobertaBase
    bpe = Factory_BPE_Pythonic(files=BPE50k_RobertaBase()).buildTokeniser()

    viz = BpeVisualiser(bpe)
    tokens, latex = viz.prepareAndTokenise_visualised_animated("horseshoe")

    print(viz.getLatexPreamble())
    print(latex)
    print(tokens)


# def makeBPE1():
#     """
#     One way to load a BPE tokeniser is to do all of it manually with bpe_knockout's language system.
#     This one produces the iconic horses/hoe.
#     """
#     from tktkt.models.bpe.base import ClassicBPE
#
#     bpe = ClassicBPE(
#         preprocessor=None,
#         vocab=vocab_and_merges.loadVocabulary(),
#         merges=vocab_and_merges.loadMerges(),
#     )
#     return bpe


def makeBPE2():
    """
    Another way is to load a HuggingFace checkpoint, and convert the resulting object into a TkTkT object.
    """
    from tktkt.factories.tokenisers import Factory_BPE_Pythonic
    from tktkt.factories.deserialisation import BPE50k_RobertaBase
    return Factory_BPE_Pythonic(files=BPE50k_RobertaBase()).buildTokeniser()


def makeBPE3():
    """
    Yet another way is to load from a TkTkT checkpoint.
    """
    from tktkt.models.bpe.base import ClassicBPE
    return ClassicBPE.from_pretrained_tktkt(checkpoint="Bauwens/RoBERTa-nl_BPE-knockout_30k")


def makeBPE4():
    """
    And lastly, you can just use a factory.
    """
    from tktkt.factories.tokenisers import Factory_BPE
    return Factory_BPE().buildTokeniser()


def makeBPEknockout():
    """
    All the visualisation code also works for BPE-knockout, since it runs on the exact same backend.
    """
    from tktkt.factories.tokenisers import Factory_BPEKnockout
    return Factory_BPEKnockout().buildTokeniser()


if __name__ == "__main__":
    from tktkt.visualisation.bpe.trees import BpeVisualiser
    bpe = makeBPEknockout()
    viz = BpeVisualiser(bpe)
    tokens, latex = viz.prepareAndTokenise_visualised_animated("horseshoe")
    print(latex)
    print(tokens)
