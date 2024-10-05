# TODO
- We have the quite large technical debt of having `Vocab` be a simple dictionary. It should really be an object that
  can, for example, support special tokens *even if* they alias the subword vocabulary (because special token strings are
  purely for visualisation; they could all be empty strings... all that matters is that they have their own ID and hence embedding).
- Preprocessors need a canonical way of accessing the boundary marker if present, the byte-based alphabet if present, etc...
- BPE-breakdown:
  - Rather than randomly selecting the two-token decomposition out of all available ones, choose it
    deterministically like Koehn & Knight 2003 with a frequency product. :eyes:
