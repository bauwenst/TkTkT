from transformers import RobertaTokenizer

from src.tktkt.preparation.spacemarking import ROBERTA_SPACING
from src.tktkt.preparation.splitters import WordSplitter
from src.tktkt.models.viterbi.framework import ViterbiTokeniser, ViterbiObjective
from src.tktkt.models.viterbi.accumulators import Plus
from src.tktkt.models.viterbi.objectives_guided import *

baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
vocab = baseline.get_vocab()

# Set up probabilities
checkpoint = "google/canine-c"  # By using a checkpoint that wasn't trained on tokenisation, you'll get random boundary probabilities, so all this doesn't say much.
probability_model_tk   = CanineTokenizer.from_pretrained(checkpoint)
probability_model_core = CanineForTokenClassification.from_pretrained(checkpoint)
######################

probability_model = HuggingFaceCharacterModelForTokenClassification(characters_to_model_input=probability_model_tk, for_token_classification=probability_model_core)
generator = MaximiseSplitsOnBoundaries(probability_model)
generator = ConstrainVocabulary(generator, subword_vocabulary=vocab, reset_value=-INFTY)  # -INFTY because they're log probabilities.
accumulator = Plus()
objective = ViterbiObjective(initial_score=0, score_generator=generator, score_combiner=accumulator)

tk = ViterbiTokeniser(
    WordSplitter(ROBERTA_SPACING),
    [objective],
    max_stepsize=15
)

words = [" flatscreentelevisie"]
for word in words:
    print(word)
    print("\tRobBERT BPE:", baseline.tokenize(word))
    print("\tSame vocab, new inference:", tk.prepareAndTokenise(word))
