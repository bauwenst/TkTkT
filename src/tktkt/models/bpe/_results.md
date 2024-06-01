# Evaluation results
The following holds for English derivational lemma morphology.

## Guided BPE-dropout
With non-deterministic guided dropout, you get mediocre results exactly like BPE-dropout gives mediocre results over BPE
(because it is a regularisation method, not a morphologically informed method):
```
GuidedBPEDropout(non-deterministic)
    Precision: 0.5203102480467269
    Recall:    0.7698372615039282
    F1:        0.6209432852034581
```

With deterministic guided dropout -- when CANINE says a position is >50% likely to be a boundary, *never* merge on it -- you get
better results:
```
    Precision: 0.5345518977897976
    Recall:    0.8041526374859708
    F1:        0.6422049184919612
```
This is better than BPE-knockout (53%, 75%, 62%) due to the improved recall caused by two effects:
1. Guided dropout actually completely disables merges at a certain boundary, rather than just maybe, see e.g. how 
`bru+ids` just becomes `bru+id+s` in BPE-knockout and hence still fails to recall the `id|s` split.
2. Unlike BPE-knockout, disabling merges is decided on a word-by-word basis.

Interestingly, the recall is much lower than just following CANINE outright and applying a Viterbi (87% recall). 
That might be due to the 50% threshold, however, since we currently pretend like a solid non-zero signal of say 30%
is exactly the same as 0%. In the dropout framework, there are no paths over probabilities, only decisions of whether or
not to apply a merge.

Let's help CANINE by making it more confident about predictions. Instead of putting a hard boundary after 50% probability,
let's lower the requirement. This will cause an increase in recall guaranteed.
```
GuidedBPEDropout(deterministic, 𝜃=0.5)
    Precision: 0.5345518977897976
    Recall:    0.8041526374859708
    F1:        0.6422049184919612

GuidedBPEDropout(deterministic, 𝜃=0.4)
    Precision: 0.5336692707861299
    Recall:    0.8260942760942761
    F1:        0.6484379301611074

GuidedBPEDropout(deterministic, 𝜃=0.3)
    Precision: 0.5304408923239066
    Recall:    0.8473063973063973
    F1:        0.6524359943826293

GuidedBPEDropout(deterministic, 𝜃=0.2)
    Precision: 0.5224756029731507
    Recall:    0.8697811447811448
    F1:        0.6528098050983984

GuidedBPEDropout(deterministic, 𝜃=0.1)
    Precision: 0.5062043390754505
    Recall:    0.8962401795735129
    F1:        0.6469855480499488
```
Highest F1 score happens when all boundaries with probability 20% and up are respected by BPE.
Precision drops monotonically, which is not surprising.

All-in-all, it's just slightly better than BPE-knockout.