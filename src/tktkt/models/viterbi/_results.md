# Evaluation results

## English morphology
To beat is BPE-knockout, which does Pr=53%, Re=75%, F1=62% on morphs and .

### BPE vocabulary
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
Using **prefix rewards** causes even worse performance, with particularly the recall being absolutely terrible.
```
    Precision: 0.43075429390415865
    Recall:    0.5206482257613858
    F1:        0.47145441435059265
```
Possibly a tiebreaker objective is needed, or perhaps there is too much reward given to matching the largest prefix
and nothing else.

Using the same rewards except **extended** past the next boundary, rather than dropping to 0 after it, gives the same
exact scores. Weird.

Using **prefix rewards with punishment for bad steps** increases all metrics.
```
		Precision: 0.49686264822134385
		Recall:    0.5619726180497345
		F1:        0.5274157598007081
```
The **extended** version is the same again!

Using **prefix rewards without punishment** and with an **AtLeastAll vocabulary constraint** boosts the precision but 
drops recall:
```
		Precision: 0.503370786516854
		Recall:    0.5006985191394244
		F1:        0.5020310967922678
```
Using the version **with punishment**, **extended** (although apparently that doesn't matter), and with 
**AtLeastAll constraint**, we get... the same thing?! How does this keep happening to these objectives?!
Even if you take out the "extended" part, it performs the exact same. So, weirdly, when you give it too much freedom,
it doesn't matter that you have punishment nor the ability to jump over the next boundary.

**TODO: Yeah so I just printed the prefix score grid and it is filled with 0s... So what the fuck have we been
testing this whole time???**

#### Unguided
Just using **least-token** Viterbi with ULM vocabulary does better:
```
    Precision: 0.5097098186675902
    Recall:    0.5756915339480302
    F1:        0.5406951569942136
```
