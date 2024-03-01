# Evaluation results
## English morphology
Running the ALBERT-v2 KudoPiece tokeniser on CELEX-en gives:

Precision: 0.5353394153423583
Recall:    0.6099189717798268
F1:        0.5702008724499125

WW-Precision: 0.1481018246027075
WW-Recall:    0.8742038216560509
WW-F1:        0.2532925090176999

This is on-par with English BPE-knockout (which has seen the test set) which sits at
at 53%, 75%, 62% (KudoPiece has +1%, -14%, -5%) and 12%, 89%, 21% (KudoPiece has +3%, -2%, +4%).

It outperforms English CANINE+BPE-Viterbi in whole-word precision and F1, but not in morph performance.
