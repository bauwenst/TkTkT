- BPE-dropout except top-down: you first do BPE entirely and then you recursively break tokens into ANY pair of tokens that concatenate to form it, NOT just the one merge BPE actually DID.
  - Extension of this: rather than randomly selecting the two-token decomposition out of all available ones, choose it
    deterministically like Koehn & Knight 2003 with a frequency product. :eyes:
