from tktkt.factories.artifacts import BPE50k_RobertaBase
from tktkt.factories.tokenisers import Factory_BPE
from tktkt.models.random.generationbased import RandomVocabSegmentation_GenerateAll, indicesToTokens, generateSegmentationIndices_exponentialSpace
from tktkt.factories.preprocessors import IdentityPreprocessor


def test_word():
    word = "reanimatie"
    vocab = {"re": 0, "anim": 1, "atie": 2, "reanim": 3, "at": 4, "ie": 5, "a": 6, "ean": 7, "r": 8}
    tk = RandomVocabSegmentation_GenerateAll(IdentityPreprocessor, vocab)

    print([indicesToTokens(word, seg) for seg in generateSegmentationIndices_exponentialSpace(word, vocab)])
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))


def test_segmentationAmount():
    from tktkt.evaluation.fertility import preprocessThenCountValidSegmentations
    from modest.languages.english import English_Celex

    tk = Factory_BPE(files=BPE50k_RobertaBase()).buildTokeniser()
    prep = tk.preprocessor
    for obj in English_Celex().generate():
        word = obj.word
        print(word, len(word))
        assert len(generateSegmentationIndices_exponentialSpace(word, tk.vocab)) == preprocessThenCountValidSegmentations(word, prep, tk.vocab)


def test_markov_forwardbackward_equivalence():
    """
    Asserts two hypotheses:
        1. Backwards and forwards Markov decoding both construct the same uniform distribution across tokenisations of
           the same string if there isn't a bias transformation (equivalent to PowerNorm with temperature 1.0).
        2. Backwards and forwards Markov deviate from each other when applying any bias transformation.
    """
    from transformers import AutoTokenizer

    from tktkt.factories.preprocessors import ModernEnglishPreprocessor, RobertaSpaceMarker
    from tktkt.models.random.grampa import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation

    vocab = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base").get_vocab()

    backwards = RandomVocabSegmentation_GreedyMarkov(
        preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),
        vocab=vocab,
        minimal_token_length=1,
        decode_backwards=True,
        probabilities_to_probabilities=PowerNormalisation(temperature=1.0)
    )
    forwards = RandomVocabSegmentation_GreedyMarkov(
        preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),
        vocab=vocab,
        minimal_token_length=1,
        decode_backwards=False,
        probabilities_to_probabilities=PowerNormalisation(temperature=1.0)
    )

    tokens = ["Ġaan", "sprak", "e", "lijk", "he", "ids", "verzekering"]

    for temperature in [1.0, 2.0, 3.0, 4.0, 5.0]:
        backwards.renormalisation.tau = temperature
        forwards.renormalisation.tau  = temperature

        print("Temp:", temperature)
        print("\tBackwards:", backwards.getJointProbability(tokens))
        print("\t Forwards:", forwards.getJointProbability(tokens))


def test_rejection_graph():
    """
    Tests the graph-based uniform segmentation sampler by Cognetta (2024).
    """
    from tktkt.models.random.rejectionsampling import RandomVocabSegmentation_RejectionSampling_UniformGraph
    from tktkt.factories.artifacts import KudoPiece32ki_SlimPajama3M

    vocab = KudoPiece32ki_SlimPajama3M()
    t = RandomVocabSegmentation_RejectionSampling_UniformGraph(
        preprocessor=vocab.preprocessorEffective(),
        vocab=vocab.getVocabulary()
    )

    from datasets import load_dataset
    corpus = load_dataset("allenai/c4", "en", streaming=True)["train"]
    for example in corpus:
        print(t.prepareAndTokenise(example["text"]))


if __name__ == "__main__":
    # test_segmentationAmount()
    test_markov_forwardbackward_equivalence()
