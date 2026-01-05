# TODO
- Known issue: `SpecialsExtended[WithSpecials]` doesn't actually detect its type variable. `SpecialsExtended` should
  probably be removed for that reason.
- Add as morphological tokenisation metrics: 
  - complete token match
  - complete affix token match
  - complete stem token match
- `Vocab`:
  - Truncating and padding to a desired size.
  - Add a name field to `Vocab`, since many tokenisers have the vocabularisation algorithm as a hyperparameter. 
  - Right now, the knowledge of how to construct/load a `Vocab` lives in both `Vocabulariser._load` as well as `AutoVocab`.
    In a sense, `AutoVocab` is a lot like `Vocabulariser.load` but for arbitrary HF tokenisers.
  - The assumption right now is that specials are only added after converting to IDs, agnostic about the content of the strings
    those IDs represent. But actually, this is not quite right. E.g.: if you use `[SPACE]` tokens to represent word boundaries,
    or `⇧` to represent capitals, you do not want the user to be able to input those strings yet you need to work in string
    space to know when to put them. This problem is avoided if you use a universal alphabet and your control/signal/special/... tokens 
    are represented entirely by characters not in that alphabet, e.g. neither `"⇧"` nor `"▁"` are pseudo-bytes.
- Nested tokeniser that protects some substrings from being segmented
  knowing that they exist fully in the vocabulary. Kind of like a pretokeniser except it only passes certain pretokens through
  (in fact, arguably, this is something that should be done before most pretokenisation) and aware of the tokeniser that follows it (I guess).
  - A good example for why you want this is phone numbers and other PII, which is part of security.
  - Another good example is the words "i.e." and "e.g." which will be destroyed by punctuation pretokenisers.
- Acceleration: tokenisation of a dataset can be done in many threads. Would greatly help the slow tokenisers to be practical.
  - Refactor from one of Vilém Zouhar's scripts:
    ```python
    def encode(self, corpus: list[str]) -> list[str]:
        import multiprocess.pool as mpl
        with mpl.Pool() as pool:
            out = pool.map(self.prepareAndTokenise, streamProgress(corpus))
        return out
    ```
- BPE-breakdown:
  - Rather than randomly selecting the two-token decomposition out of all available ones, choose it
    deterministically like Koehn & Knight 2003 with a frequency product. :eyes:
