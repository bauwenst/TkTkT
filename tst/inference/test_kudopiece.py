from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.preparation.instances import IdentityMapper, AppendSpace, IdentityPretokeniser, Preprocessor
from tktkt.paths import TkTkTPaths

prep = Preprocessor(IdentityMapper(), AppendSpace(front_not_back=True), IdentityPretokeniser())
tk = KudoPieceTokeniser(prep, TkTkTPaths.pathToModels() / "kudopiece_en" / "kudopiece_en_2024-02-27_16-09-37.model")

# s = " This is an example sentence."
s = " a to of in is I on it as be or at by an we my A To Of In Is On It As Be Or At By An We My"
s = " ௲"
print(tk.tokenise(s))
print(list(tk.vocab.keys()))


def fst(tuple):
    return tuple[0]


from transformers import AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast
tk2: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")

# print(len(tk.get_vocab()), sorted(tk2.get_vocab().items(), key=lambda i: i[1]))
print(len(tk2.get_vocab()), list(map(fst, sorted(tk2.get_vocab().items(), key=lambda i: i[1]))))
print(tk2.tokenize(s))  # Strange behaviour: it produces [CLS, _, UNK, SEP] and yet .tokenize outputs the string itself.
# print(tk2.tokenize("To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator."))
