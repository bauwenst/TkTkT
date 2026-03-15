from tst.preamble import *
from tktkt.models.kudopiece.vocabularisation import *
from tktkt.factories.preprocessors import ModernEnglishPreprocessor_SentencePieceCompatible


def main(tsv: Path):
    trainer = KudoPieceVocabulariser(
        preprocessor=ModernEnglishPreprocessor_SentencePieceCompatible(BoundaryMarkerLocation.START),
        final_vocab_size=40_000,
        arguments=KudoPieceArguments(character_coverage=0.9995),
        file_stem="kudopiece_" + tsv.stem
    )
    trainer.vocabulariseFromTsv(tsv)


if __name__ == "__main__":
    main(...)   # We don't have any word-count files :(
