# Evaluation results

## English morphology
To beat is BPE-knockout, which does Pr=53%, Re=75%, F1=62% on morphs and .

### BPE vocabulary
#### Probability-based
**CANINE with boundary-only symmetric probabilities** (2*P-1) and English BPE vocabulary as constraint:
```
    Precision: 0.5707395498392283
    Recall:    0.8034367141659682
    F1:        0.6673861579167247
```
which is respectively +4%, +5%, +4% on BPE-knockout.

The vocabulary does matter a lot. The character model by itself does 92% on all metrics, and if you switch to a Dutch
vocabulary (RobBERT's) with the same objective, you get
```
    Precision: 0.43320889909887733
    Recall:    0.8851913942442023
    F1:        0.5817243690381102
```
which is respectively -10%, +13% and -4% over English BPE-knockout, clearly oversegmenting into small tokens because the
bigger meaningful tokens are unknown to it.

We have three model alternatives:
    - Symmetric probabilities but using the joint (i.e. also counting the non-boundaries);
    - Boundary-only log probabilities
    - Joint log probabilities
Strangely, the first one changes 0 of the predicted splits.

When you use log probabilities WITHOUT the joint, performance tanks.
```
    Precision: 0.5128729963008631
    Recall:    0.5810841017043867
    F1:        0.5448519779931884
    WW-Precision: 0.149173859432799
    WW-Recall:    0.8756514186450493
    WW-F1:        0.2549201399131864
```

Using joint log probabilities, which is probabilistically the most sound, actually produces the best tokeniser so far!
```
    Precision: 0.5703759458067583
    Recall:    0.8045822855546242
    F1:        0.6675321062636191
```
...with whole-word metrics beating BPE-knockout (12%, 89%, 21%) everywhere too:
```
    WW-Precision: 0.12900606108624174
    WW-Recall:    0.9428199189345686
    WW-F1:        0.22695752169216296
```

#### Unguided
**Token-minimising Viterbi** performs markedly worse, with (no surprise) way worse recall:
```
    Precision: 0.4559803399964531
    Recall:    0.5028778988544286
    F1:        0.47828224445595985
```
In fact, its recall is so low that despite trying to find the biggest units possible, it underperforms BPE-knockout
in whole-word boundary recall by 5%:
```
    WW-Precision: 0.1464619594132401
    WW-Recall:    0.8368558193398957
    WW-F1:        0.249293861445913
```

### ULM vocabulary
#### Probability-based
Using **CANINE with symmetric probability** (joint or not) and a ULM vocabulary:
```
    Precision: 0.583957433992571
    Recall:    0.8126292260407936
    F1:        0.6795724049301947

    WW-Precision: 0.13169360505973296
    WW-Recall:    0.9494788650839606
    WW-F1:        0.2313049918008217
```
...which is even better than anything BPE had, confirming that the vocab is the bottleneck.

The joint log probability has a lower precision and F1 this time:
```
    Precision: 0.5824413277045277
    Recall:    0.8133836267113719
    F1:        0.6788075223560411
```

And again, scores are worst for boundary-only log probabilities:
```
    Precision: 0.5452460090819718
    Recall:    0.6508521933500978
    F1:        0.5933869981658856
```

Using the **AtLeastAll constraint with symmetric probability objective** unsurprisingly gives a result that is
at least as good as with an exact vocabulary:
```
		Precision: 0.6577800897327525
		Recall:    0.8479463537300922
		F1:        0.7408546632978139

		WW-Precision: 0.14153499360599953
		WW-Recall:    0.9452808338158657
		WW-F1:        0.2462060514657366
```
Indeed, this blows the above 58%, 81%, 68% of the exact constraint out of the water, at the cost of concatenability. 
Precision could be better though; seems to be oversegmenting (finds 84% of real splits but only 66% of proposed splits are real).

Now let's switch back to the Exact constraint but use a different probability-to-score transform:
```
NO EQUIVALENCE:

HFPointViterbi(BoundaryScoresChosen(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

HFPointViterbi(BoundaryScoresAll(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5839858651568084
    Recall:    0.8126851075719475
    F1:        0.6796111967848965  ---> NEW BEST!

---------------------------------------
YES EQUIVALENCE (which implies that with this objective, you can't actually take joint effect into account):
Also equivalent to the first LinearPT above.

HFPointViterbi(BoundaryScoresChosen(LinearPT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

HFPointViterbi(BoundaryScoresAll(LinearPT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

---------------------------------------
NO EQUIVALENCE:

HFPointViterbi(BoundaryScoresChosen(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5861179889091244
    Recall:    0.8003073484213468
    F1:        0.6766675722604802

HFPointViterbi(BoundaryScoresAll(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5839858651568084
    Recall:    0.8126851075719475
    F1:        0.6796111967848965

---------------------------------------
YES EQUIVALENCE:
(Again equivalent to the first class above.)

HFPointViterbi(BoundaryScoresChosen(PiecewisePT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.5861179889091244
    Recall:    0.8003073484213468
    F1:        0.6766675722604802

HFPointViterbi(BoundaryScoresAll(PiecewisePT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.5861179889091244
    Recall:    0.8003073484213468
    F1:        0.6766675722604802
```
The conclusion is that
1. Taking into account the point where you didn't split is useless *if* the reward you get for not splitting at a position
   is the *opposite* of what you get for splitting there, *either* because you force it to be (like we do above) or when
   the score of the complement is the opposite by accident (e.g. for all symmetric transforms, which we already knew).
2. It is useless to test such reward structures *at all*, because apparently, using the negative score in BoundaryScoresChosen
   and in BoundaryScoresAll is equivalent to using the score of the complement in BoundaryScoresChosen.

That means we no longer need to consider NegComp, and we now also know that symmetric transforms don't benefit from
considering all boundaries.

#### Hard-boundary-based
For exact vocabulary constraint, we are looking to beat 58%, 81%, 68%:
```
HFPointViterbi(HardBoundaryPrefixLength + VocabularyConstraintExact)
    Precision: 0.5290765557743583
    Recall:    0.8482816429170159
    F1:        0.6516909405085165
    WW-Precision: 0.11468553404318352
    WW-Recall:    0.9526635784597568
    WW-F1:        0.2047253892457731

HFPointViterbi(HardBoundaryPrefixLengthExtended + VocabularyConstraintExact)
    Precision: 0.5411118078866797
    Recall:    0.8197261804973456
    F1:        0.6518976091014131
    WW-Precision: 0.12114058061898263
    WW-Recall:    0.950781702374059
    WW-F1:        0.2149003697281026

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLength + VocabularyConstraintExact)
    Precision: 0.5359700205844579
    Recall:    0.8511874825370215
    F1:        0.6577637672867028
    WW-Precision: 0.11627579654814477
    WW-Recall:    0.9567168500289519
    WW-F1:        0.2073509341616076

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLengthExtended + VocabularyConstraintExact)
    Precision: 0.5541110599785243
    Recall:    0.8074322436434759
    F1:        0.6572058856973915
    WW-Precision: 0.1262080073630925
    WW-Recall:    0.9528083381586566
    WW-F1:        0.22289197426346088
```
For at-least constraint, we are looking to beat 66%, 85%, 74%:
```
HFPointViterbi(HardBoundaryPrefixLength + VocabularyConstraintAtLeastAll)
    Precision: 0.6116584912043301
    Recall:    0.8082984073763622
    F1:        0.6963628048046602
    WW-Precision: 0.1357620094722598
    WW-Recall:    0.9295020266357846
    WW-F1:        0.23691978451774773

HFPointViterbi(HardBoundaryPrefixLengthExtended + VocabularyConstraintAtLeastAll)
    Precision: 0.6116584912043301
    Recall:    0.8082984073763622
    F1:        0.6963628048046602
    WW-Precision: 0.1357620094722598
    WW-Recall:    0.9295020266357846
    WW-F1:        0.23691978451774773

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLength + VocabularyConstraintAtLeastAll)
    Precision: 0.6199262778854964
    Recall:    0.8129365744621403
    F1:        0.7034319354955697
    WW-Precision: 0.13787739969744103
    WW-Recall:    0.9367400115807759
    WW-F1:        0.24037443583885884

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLengthExtended + VocabularyConstraintAtLeastAll)
    Precision: 0.620031118784236
    Recall:    0.8127968706342554
    F1:        0.7034471084672398
    WW-Precision: 0.1378604770125967
    WW-Recall:    0.9363057324840764
    WW-F1:        0.24033441709242917
```

#### Unguided
Just using **least-token** Viterbi with ULM vocabulary does better:
```
    Precision: 0.5097098186675902
    Recall:    0.5756915339480302
    F1:        0.5406951569942136
```
