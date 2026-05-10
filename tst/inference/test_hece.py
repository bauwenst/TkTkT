from tst.preamble import *

from tktkt.factories.preprocessors import *
from tktkt.models.linguistic.hece import Hece


tk = Hece(preprocessor=Preprocessor(
    uninvertible_mapping=Lowercaser(),
    splitter=TraditionalPretokeniser()
))

test_words = [
    "türkiye", "kardeş", "trabzon",
    "matematikçiler", "atasözleri", "geçmişten", "günümüze",
]
# Expected results:
#   hecele("türkiye") == ['tür', 'ki', 'ye']
#   hecele("kardeş") == ['kar', 'deş']
#   hecele("trabzon") == ['t', 'rab', 'zon']

print("=" * 40)
for word in test_words:
    syllables = tk.prepareAndTokenise(word)
    print(f"{word:20s} -> {' + '.join(syllables)}")

print()
text = "Atasözleri geçmişten günümüze kadar ulaşan sözlerdir"
print(f"Input : {text}")
print(f"Output: {tk.prepareAndTokenise(text)}")
