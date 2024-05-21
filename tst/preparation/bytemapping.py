from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from tktkt.preparation.mappers import PseudoByteMapping


def assert_bytemap_equality():
    assert PseudoByteMapping.bytes_to_unicode_documented() == bytes_to_unicode()
    assert PseudoByteMapping.bytes_to_unicode_softcoded() == bytes_to_unicode()


if __name__ == "__main__":
    assert_bytemap_equality()
