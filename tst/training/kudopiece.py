from tst.preamble import *
from tktkt.models.kudopiece.training import *

from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, KnockoutDataConfiguration, setupEnglish
from string import ascii_letters


def main():
    args_alpha = KudoPieceArguments_Alphabet(required_chars=[l for l in ascii_letters], byte_fallback=True, character_coverage=0.9995)
    args_algo = KudoPieceArguments_Algorithm()

    with KnockoutDataConfiguration(setupEnglish()):
        trainer = KudoPieceTrainer(
            word_boundary_location=BoundaryMarkerLocation.START,
            final_vocab_size=40_000,
            alphabet_arguments=args_alpha,
            algorithm_arguments=args_algo,
            file_stem="kudopiece_" + Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()
        )
        trainer.train_from_wordfile(Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_counts)


if __name__ == "__main__":
    main()