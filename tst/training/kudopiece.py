from tktkt.models.kudopiece.training import *

from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯
from string import ascii_letters

args_alpha = KudoPieceArguments_Alphabet(required_chars=[l for l in ascii_letters], byte_fallback=True,
                                         character_coverage=1.0)
args_algo = KudoPieceArguments_Algorithm()

trainer = KudoPieceTrainer(final_vocab_size=8000, alphabet_arguments=args_alpha, algorithm_arguments=args_algo)
trainer.train_from_wordfile(Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_counts)
