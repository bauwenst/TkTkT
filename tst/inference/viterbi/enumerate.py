from tktkt.models.random.generationbased import generateSegmentationIndices_fixedSpace
from tktkt.util.strings import indicesToTokens

from tktkt.factories.deserialisation import BPE32ki_SlimPajama3M

s = "accumulatively"

english_bpe = BPE32ki_SlimPajama3M()
preprocessor = english_bpe.preprocessorEffective()

s = preprocessor.do(s)[0]
all_segmentations = generateSegmentationIndices_fixedSpace(s, english_bpe.buildVocabulary())

print("Enumerating...")
print(
    list(
        sorted(
            filter(lambda segmentation: not any(len(t) == 1 for t in segmentation[1:]),
                   map(lambda idcs: indicesToTokens(s, idcs),
                       all_segmentations)), reverse=True)))
