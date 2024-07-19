import pytest
from tokenizer.tokenizer.encoding_decoding import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def text_to_encode():
    return "Praise the sun"


@pytest.fixture
def text_to_decode():
    return [397, 32, 342, 259, 32, 547, 32, 84, 859, 32, 79, 456, 32, 80, 291, 519, 32, 304, 32, 79, 98, 115,
            264, 118, 618]


def test_encode(tokenizer, text_to_encode):
    encoded_text = tokenizer.encode(text_to_encode)
    assert isinstance(encoded_text, list)
    assert isinstance(encoded_text[0], int)


def test_decode(tokenizer, text_to_decode):
    decoded_text = tokenizer.decode(text_to_decode)
    assert isinstance(decoded_text, str)


def test_composition_identity(tokenizer, text_to_encode):
    # An encoded string when decoded must return to itself
    assert tokenizer.decode(tokenizer.encode(text_to_encode)) == text_to_encode
