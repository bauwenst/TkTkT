"""
For documentation purposes only, the Python equivalent of
https://github.com/google/sentencepiece/blob/master/src/model_interface.cc#L153
"""
from typing import List
from dataclasses import dataclass

@dataclass
class View:
    span_start: int
    span_length: int


def splitIntoWords(text: str, whitespace_suffix_not_prefix: bool, allow_whitespace_pieces: bool) -> List[View]:
    # Pointers to the start and end of the input
    begin = 0
    end   = begin + len(text)

    # Space symbol (U+2581)
    kSpaceSymbol = b"\xe2\x96\x81".decode("utf-8")
    in_whitespace_sequence = False

    result = []
    if whitespace_suffix_not_prefix:  # put ws tokens at the end of non-ws sequences.
        if begin < end:
            result.append(View(begin, 0))

        while begin < end:
            mblen = min(len(text[begin]), end - begin)
            is_whitespace = text[begin:begin+mblen] == kSpaceSymbol

            if is_whitespace:  # keep track of sequences consecutive ws tokens.
                in_whitespace_sequence = True
            elif in_whitespace_sequence:  # you were in whitespace, but you've hit a letter.
                if allow_whitespace_pieces:
                    result.append(View(begin, 0))

                in_whitespace_sequence = False

            result[-1] = View(result[-1].span_start, result[-1].span_length + mblen)
            begin += mblen

            if begin < end and is_whitespace and not allow_whitespace_pieces:
                result.append(View(begin, 0))

    else:

        while begin < end:
            mblen = min(len(text[begin]), end - begin)
            is_whitespace = text[begin:begin+mblen] == kSpaceSymbol

            # if is whitespace (and not in sequence if allow_ws_only_pieces is True)
            if begin == 0 or (is_whitespace and (not in_whitespace_sequence or not allow_whitespace_pieces)):
                result.append(View(begin, 0))  # add empty string piece.
                in_whitespace_sequence = True

            if in_whitespace_sequence and not is_whitespace:
                in_whitespace_sequence = False

            result[-1] = View(result[-1].span_start, result[-1].span_length + mblen)
            begin += mblen

    return result


def tokeniseFromViews(text: str, views: List[View]) -> List[str]:
    return [text[view.span_start:view.span_start+view.span_length] for view in views]


s = "This is a    test sentence.".replace(" ", b"\xe2\x96\x81".decode("utf-8"))
settings = [(False, False), (False, True), (True, False), (True, True)]
for e1, e2 in settings:
    print(tokeniseFromViews(s, splitIntoWords(s, whitespace_suffix_not_prefix=e1, allow_whitespace_pieces=e2)))
