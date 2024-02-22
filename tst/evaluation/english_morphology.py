from transformers import CanineTokenizer, CanineForTokenClassification

from bpe_knockout.project.config import TemporaryContext, setupEnglish
from bpe_knockout.auxiliary.robbert_tokenizer import robbert_tokenizer

from tktkt.preparation.instances import RobertaPreprocessor
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.viterbi.instances import HFModelViterbi, LeastTokenViterbi
from tktkt.files.paths import relativeToCwd

from tst.preamble import *


TOKENISER = HFModelViterbi(
    RobertaPreprocessor,
    vocab=robbert_tokenizer.get_vocab(),
    max_step=20,
    huggingface_checkpoint=relativeToCwd(checkpoints_path / "CANINE-C_2024-02-12_19-35-28").as_posix(),
    tokeniser_class=CanineTokenizer,
    model_class=CanineForTokenClassification
)


with TemporaryContext(setupEnglish()):
    intrinsicEvaluation([TOKENISER], do_whole_word=True, verbose=True)
