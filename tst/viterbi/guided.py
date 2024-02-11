from src.tktkt.preparation.spacemarking import ROBERTA_SPACING
from src.tktkt.preparation.splitters import WordSplitter
from src.tktkt.models.viterbi.framework import ViterbiTokeniser, ViterbiObjective
from src.tktkt.models.viterbi.accumulators import Plus
from src.tktkt.models.viterbi.objectives_guided import HuggingFaceCharacterModelForTokenClassification, CanineTokenizer, CanineForTokenClassification, MaximiseSplitsOnBoundaries


tk  = CanineTokenizer.from_pretrained("google/canine-c")
mod = CanineForTokenClassification.from_pretrained("google/canine-c")

word = " flatscreentelevisie"

probability_model = HuggingFaceCharacterModelForTokenClassification(characters_to_model_input=tk, for_token_classification=mod)
generator = MaximiseSplitsOnBoundaries(probability_model)
accumulator = Plus()
objective = ViterbiObjective(initial_score=0, score_generator=generator, score_combiner=accumulator)

tk = ViterbiTokeniser(
    WordSplitter(ROBERTA_SPACING),
    [objective],
    max_stepsize=15
)

print(tk.prepareAndTokenise(word))