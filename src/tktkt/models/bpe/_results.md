# Evaluation results
The following holds for English derivational lemma morphology in `bpe_knockout` with the `legacy = True` option
(i.e. these evaluations include apostrophised words).

## BPE-knockout
```
BPE-knockout:
    Precision: 0.5293550355252022
    Recall:    0.7514948309583682
    F1:        0.6211619071813761
```
Note: results from the paper differ slightly due to it using `legacy = False` (ironically). With that option, you get
```
BPE-knockout (no apostrophised words):
    Precision: 0.5324047358471794
    Recall:    0.7507295173961841
    F1:        0.6229931893591012
```
which matches the paper exactly.

## ReBPE
ReBPE (reifying BPE) takes a BPE-knockout tokeniser and tries to turn it back 
into a BPE tokeniser. This can be done iteratively.

```
ReBPE-1 (knockout-reify-knockout):
	Precision: 0.5453513648692841
    Recall:    0.7920927633417155
    F1:        0.645961742221412

ReBPE-2 (knockout-reify-knockout-reify-knockout):
    Precision: 0.5488506296218447
    Recall:    0.8025426096675049
    F1:        0.6518842980833608

ReBPE-3:
    Precision: 0.5497135217723453
    Recall:    0.8042190556021235
    F1:        0.6530459444129325

ReBPE-4:
    Precision: 0.5497441185456767
    Recall:    0.8043867001955853
    F1:        0.6531228022414302

ReBPE-5:
    Precision: 0.5497441185456767
    Recall:    0.8043867001955853
    F1:        0.6531228022414302
```
We see recall increase by +5% while precision also rises by over +1.5%.

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