from tktkt.builders.english import getEnglishBpeFiles
from tktkt.models.random.generationbased import generateSegmentationIndices_fixedSpace
from tktkt.preparation.huggingface import HuggingFacePreprocessorForWords
from tktkt.util.strings import segmentUsingIndices

s = "accumulatively"

english_bpe = getEnglishBpeFiles()
preprocessor = HuggingFacePreprocessorForWords(english_bpe.toFastBPE())

s = preprocessor.do(s)[0]
all_segmentations = generateSegmentationIndices_fixedSpace(s, english_bpe.loadVocabulary())

print("Enumerating...")
print(
    list(
        sorted(
            filter(lambda segmentation: not any(len(t) == 1 for t in segmentation[1:]),
                   map(lambda idcs: segmentUsingIndices(s, idcs),
                       all_segmentations)), reverse=True)))
