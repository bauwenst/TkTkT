from tst.preamble import *
from tktkt.models.kudopiece.vocabularisation import *
from tktkt.factories.preprocessing import ModernEnglishPreprocessor_SentencePieceCompatible

from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, KnockoutDataConfiguration, setupEnglish


def main():
    with KnockoutDataConfiguration(setupEnglish()):
        trainer = KudoPieceVocabulariser(
            preprocessor=ModernEnglishPreprocessor_SentencePieceCompatible(BoundaryMarkerLocation.START),
            final_vocab_size=40_000,
            arguments=KudoPieceArguments(character_coverage=0.9995),
            file_stem="kudopiece_" + Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag()
        )
        trainer.vocabulariseFromTsv(Pℛ𝒪𝒥ℰ𝒞𝒯.config.lemma_counts.path)


if __name__ == "__main__":
    main()