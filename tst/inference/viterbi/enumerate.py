from tktkt.models.viterbi.objectives_postprocessors import ConstrainVocabulary
from tst.evaluation.english_morphology import canine_viterbi

vocab_manager: ConstrainVocabulary = canine_viterbi.objectives[0].score_generator
s = "accumulatively"

print("Enumerating...")
all_segmentations = vocab_manager.getAllPossibleSegmentations(
    canine_viterbi.preprocessor.do(s)[0],
    max_k=20
)
print(list(sorted(filter(lambda segmentation: not any(len(t) == 1 for t in segmentation[1:]), all_segmentations), reverse=True)))
