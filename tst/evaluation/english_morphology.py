"""
Evaluate any tokeniser on English morphology.

    CANINE performs as follows:
		Precision: 0.20260564287554753
		Recall:    0.46133255106156723
		F1:        0.2815580695334967
    That can't be right... Something buggy is happening. There is too much intelligence in this model for it to do so poorly
    given that BPE does it much better and is dumb.
    |
    Yup. The bug is that although the Viterbi is doing everything properly, the pretokeniser undo() is stripping characters off of tokens.
    It's AppendSpace that's doing this.
    We predicted that this would happen, because this was originally applied to pretokens (append 1 space to a pretoken that
    will be split in the future) whilst now it is applied to EVERY token. Indeed, there are pretokenisation steps that
    should only be applied after re-concatenating the tokens (except you don't know how much to concatenate them... messy!).
    |
    After a quick fix, it's now
        Precision: 0.43320889909887733
		Recall:    0.8851913942442023
		F1:        0.5817243690381102
    which is respectively -10%, +13% and -4% over English BPE-knockout (53%, 75%, 62%). Clearly oversegmenting...
    Aha, but we were using RobBERT's vocabulary for Viterbi steps, which is a Dutch vocabulary, not English!
    With English BPE vocabulary:
        Precision: 0.5707395498392283
		Recall:    0.8034367141659682
		F1:        0.6673861579167247
    which is respectively +4%, +5%, +4% on BPE-knockout. Still oversegmenting, but the precision and F1 are better!

    TODO: Does this outperform BPE-knockout-reify?
    TODO: Now do it with ULM vocab.

    Token-minimising Viterbi performs markedly worse, with (no surprise) way worse recall:
		Precision: 0.4559803399964531
		Recall:    0.5028778988544286
		F1:        0.47828224445595985
    In fact, its recall is so low that despite trying to find the biggest units possible, it underperforms BPE-knockout
    in whole-word boundary recall by 5%:
		WW-Precision: 0.1464619594132401
		WW-Recall:    0.8368558193398957
		WW-F1:        0.249293861445913

TODO: There are two issues with our CANINE evaluation.
    1. I'm not sure if it got special tokens during pretraining, and it is likely not a good idea to leave them out in
       both fine-tuning and inference. The model is used to using these as working memory, most likely.
        - In fine-tuning, you would need to pad the labels based on special tokens. There has to be a function for this
          in HuggingFace because which special tokens are added is a tokeniser-specific decision that can't be predicted.
        - In inference, you would need to filter these from the prediction output before handing them to your Viterbi lattice.
    2. Because the vocabulary I am using to limit the Viterbi steps during inference is specified in pseudo-bytes, I am
       giving CANINE an input in pseudo-bytes for inference too. This is a problem, because CANINE wasn't pre-trained
       nor fine-tuned with pseudo-bytes, but with a modulo mapping.
       What this means is that ë is going to show up as Ã<< and CANINE won't know what it means since it only ever saw
       ë during training and fine-tuning.
       |
       What you have here is a problem caused by the fact that we are using a language model as tokeniser for another
       language model with different pretokenisation. Indeed, we need two pretokenisers, and one of them should be
       applied AFTER the tokeniser!
         1. Give the original text to CANINE. (It uses context to figure out where to put segmentation boundaries.)
         2. Invert the pseudo-byte mapping of the vocabulary to recognise which steps correspond to valid tokens, and give the
            result to the Viterbi tokeniser. Now you have segmentations into strings that include spaces and ë etc.
         2. Apply the byte mapping of the LM to map these tokens into the LM vocabulary.
"""
from transformers import CanineTokenizer, CanineForTokenClassification

from bpe_knockout.project.config import TemporaryContext, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯

from tktkt.preparation.instances import HuggingFacePreprocessorForWords
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.viterbi.instances import HFModelViterbi, LeastTokenViterbi
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
from tktkt.files.paths import relativeToCwd, DataPaths

from tst.preamble import *


with TemporaryContext(setupEnglish()):
    english_bpe = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.toFastBPE()  # Has a byte-based preprocessor; HuggingFace sets it automatically on all Roberta tokenisers.

english_bpe_interface = HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)

compressive_viterbi = LeastTokenViterbi(
    HuggingFacePreprocessorForWords(english_bpe),
    vocab=english_bpe.get_vocab(),
    max_step=20
)

canine_viterbi = HFModelViterbi(
    # HuggingFacePreprocessorForWords(robbert_tokenizer),  # The preprocessor that maps any string into the space of the vocabulary used.
    # vocab=robbert_tokenizer.get_vocab(),                 # The vocabulary that limits Viterbi steps.
    HuggingFacePreprocessorForWords(english_bpe),
    vocab=english_bpe.get_vocab(),
    max_step=20,
    huggingface_checkpoint=relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28").as_posix(),
    tokeniser_class=CanineTokenizer,
    model_class=CanineForTokenClassification,

    tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
)

tokenisers_to_evaluate = [
    # english_bpe_interface,
    compressive_viterbi,
    canine_viterbi,
]

if __name__ == "__main__":
    with TemporaryContext(setupEnglish()):
        intrinsicEvaluation(tokenisers_to_evaluate, do_whole_word=True, verbose=True)
