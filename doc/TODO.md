# TODO
- Add as morphological tokenisation metrics: 
  - complete token match
  - complete affix token match
  - complete stem token match
- We have the quite large technical debt of having `Vocab` be a simple dictionary. It should really be an object with:
  - A name for the vocabulary, because many tokenisers have the vocabularisation algorithm as a hyperparameter. 
  - Support for special tokens *even if* they alias the subword vocabulary (because special token strings are
    purely for visualisation; they could all be empty strings... all that matters is that they have their own ID and hence embedding).
     - One reason you should have separate sets for the subword vocab and the specials is that there are algorithms that
       rely entirely on the vocabulary strings to create segmentations. BPE uses a merge file and ULM uses a likelihood file,
       but MaxMatch, GRaMPa, ..., all use just a vocabulary of strings.
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
