"""
Tests for the language-specific preprocessors.
"""


def tst_japaneseSplitter():
    from tktkt.preparation.splitters import IntoJapaneseWords

    pretokeniser = IntoJapaneseWords()
    # Examples from the Fugashi documentation.
    print(pretokeniser.split("日本語ですよ"))  # Should be 日本語 です よ
    print(pretokeniser.split("深海魚は、深海に生息する魚類の総称。"))  # Should be 深海魚 は 、 深海 に 生息 する 魚類 の 総称 。


def tst_thaiSplitter():
    from tktkt.preparation.splitters import IntoThaiWords

    pretokeniser = IntoThaiWords()
    # Example taken from https://www.bananathaischool.com/blog/sentences-in-thai-class/
    print(pretokeniser.split("ช่วยอธิบายให้ฉันฟังหน่อย"))  # Should become ช่วย อธิบาย ให้ ฉัน ฟัง หน่อย
