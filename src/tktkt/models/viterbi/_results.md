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


#### Hard-boundary-based
~~Using **prefix rewards** causes even worse performance, with particularly the recall being absolutely terrible.~~
```
    Precision: 0.43075429390415865
    Recall:    0.5206482257613858
    F1:        0.47145441435059265
```
~~Possibly a tiebreaker objective is needed, or perhaps there is too much reward given to matching the largest prefix
and nothing else. Using the same rewards except **extended** past the next boundary, rather than dropping to 0 after it, gives the same
exact scores. Weird.~~

~~Using **prefix rewards with punishment for bad steps** increases all metrics.~~
```
		Precision: 0.49686264822134385
		Recall:    0.5619726180497345
		F1:        0.5274157598007081
```
~~The **extended** version is the same again!~~

~~Using **prefix rewards without punishment** and with an **AtLeastAll vocabulary constraint** boosts the precision but 
drops recall:~~
```
		Precision: 0.503370786516854
		Recall:    0.5006985191394244
		F1:        0.5020310967922678
```
~~Using the version **with punishment**, **extended** (although apparently that doesn't matter), and with 
**AtLeastAll constraint**, we get... the same thing?! How does this keep happening to these objectives?!
Even if you take out the "extended" part, it performs the exact same. So, weirdly, when you give it too much freedom,
it doesn't matter that you have punishment nor the ability to jump over the next boundary.~~

~~**TODO: Yeah so I just printed the prefix score grid and it is filled with 0s... So what the fuck have we been
testing this whole time???**~~

Here are the new results. For exact vocabulary constraint, we are looking to beat 58%, 81%, 68%:
```
HFPointViterbi(HardBoundaryPrefixLength + VocabularyConstraintExact)
        Morph split accuracy:
                Precision: 0.5290765557743583
                Recall:    0.8482816429170159
                F1:        0.6516909405085165
        Lemmatic split accuracy:
                Precision: 0.11468553404318352
                Recall:    0.9526635784597568
                F1:        0.2047253892457731

HFPointViterbi(HardBoundaryPrefixLengthExtended + VocabularyConstraintExact)
        Morph split accuracy:
                Precision: 0.5411118078866797
                Recall:    0.8197261804973456
                F1:        0.6518976091014131
        Lemmatic split accuracy:
                Precision: 0.12114058061898263
                Recall:    0.950781702374059
                F1:        0.2149003697281026

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLength + VocabularyConstraintExact)
        Morph split accuracy:
                Precision: 0.5359700205844579
                Recall:    0.8511874825370215
                F1:        0.6577637672867028
        Lemmatic split accuracy:
                Precision: 0.11627579654814477
                Recall:    0.9567168500289519
                F1:        0.2073509341616076

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLengthExtended + VocabularyConstraintExact)
        Morph split accuracy:
                Precision: 0.5541110599785243
                Recall:    0.8074322436434759
                F1:        0.6572058856973915
        Lemmatic split accuracy:
                Precision: 0.1262080073630925
                Recall:    0.9528083381586566
                F1:        0.22289197426346088
```
For at-least constraint, we are looking to beat 66%, 85%, 74%:
```
HFPointViterbi(HardBoundaryPrefixLength + VocabularyConstraintAtLeastAll)
        Morph split accuracy:
                Precision: 0.6116584912043301
                Recall:    0.8082984073763622
                F1:        0.6963628048046602
        Lemmatic split accuracy:
                Precision: 0.1357620094722598
                Recall:    0.9295020266357846
                F1:        0.23691978451774773

HFPointViterbi(HardBoundaryPrefixLengthExtended + VocabularyConstraintAtLeastAll)
        Morph split accuracy:
                Precision: 0.6116584912043301
                Recall:    0.8082984073763622
                F1:        0.6963628048046602
        Lemmatic split accuracy:
                Precision: 0.1357620094722598
                Recall:    0.9295020266357846
                F1:        0.23691978451774773

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLength + VocabularyConstraintAtLeastAll)
        Morph split accuracy:
                Precision: 0.6199262778854964
                Recall:    0.8129365744621403
                F1:        0.7034319354955697
        Lemmatic split accuracy:
                Precision: 0.13787739969744103
                Recall:    0.9367400115807759
                F1:        0.24037443583885884

HFPointViterbi(HardBoundaryAndNonBoundaryPrefixLengthExtended + VocabularyConstraintAtLeastAll)
        Morph split accuracy:
                Precision: 0.620031118784236
                Recall:    0.8127968706342554
                F1:        0.7034471084672398
        Lemmatic split accuracy:
                Precision: 0.1378604770125967
                Recall:    0.9363057324840764
                F1:        0.24033441709242917
```

#### Unguided
Just using **least-token** Viterbi with ULM vocabulary does better:
```
    Precision: 0.5097098186675902
    Recall:    0.5756915339480302
    F1:        0.5406951569942136
```
