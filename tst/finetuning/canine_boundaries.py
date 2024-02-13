from src.tktkt.visualisation.neural.splitpoints_probabilities import *

# Path setup
from src.tktkt.files.paths import setTkTkToutputRoot, getTkTkToutputPath, PATH_ROOT
setTkTkToutputRoot(PATH_ROOT / "data" / "out")
checkpoints_path = getTkTkToutputPath() / "checkpoints"
checkpoints_path.mkdir(parents=True, exist_ok=True)

# Classifier setup
from src.tktkt.models.viterbi.objectives_guided import HuggingFaceCharacterModelForTokenClassification, CanineTokenizer, CanineForTokenClassification
tk   = CanineTokenizer.from_pretrained("google/canine-c")
core = from_pretrained_absolutePath(CanineForTokenClassification, checkpoints_path / "CANINE-C_2024-02-12_19-35-28")
classifier = HuggingFaceCharacterModelForTokenClassification(tk, core)

# Examples
words = [" establishmentarianism", " rainbow-coloured", " superbizarre", " algebraically", " ascertainably",
         " barelegged", " behaviourism", " chauvinistically", " maladministration",
         " ethnographically", " good-neighbourliness", " heavy-handedness",
         " existentialist", " imperialistically", " materialization", " decentralization", " disadvantageously",
         " nearsightedness", " neglectfulness"]
for word in words:
    print(visualisePredictedBoundaries(classifier, word))
