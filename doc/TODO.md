# TODO
- The KudoPiece vocabulariser apparently does not split off hyphens at train time. That's because the SentencePiecePreprocessor
  has no hyphen nor digit splitting. Either set the "separate character families" option to be true, or find a way to
  sneak in any kind of pretoken (which would be the more general approach, e.g. for morpheme pretokenisers).
- We have the quite large technical debt of having `Vocab` be a simple dictionary. It should really be an object with:
  - A name for the vocabulary, because many tokenisers have the vocabularisation algorithm as a hyperparameter. 
  - Support for special tokens *even if* they alias the subword vocabulary (because special token strings are
    purely for visualisation; they could all be empty strings... all that matters is that they have their own ID and hence embedding).
     - One reason you should have separate sets for the subword vocab and the specials is that there are algorithms that
       rely entirely on the vocabulary strings to create segmentations. BPE uses a merge file and ULM uses a likelihood file,
       but MaxMatch, GRaMPa, ..., all use just a vocabulary of strings.
  - A notion of "alphabet".
- Preprocessors need a canonical way of accessing the boundary marker if present, the byte-based alphabet if present, etc...
- Acceleration: tokenisation of a dataset can be done in many threads. Would greatly help the slow tokenisers to be practical.
- BPE-breakdown:
  - Rather than randomly selecting the two-token decomposition out of all available ones, choose it
    deterministically like Koehn & Knight 2003 with a frequency product. :eyes:
