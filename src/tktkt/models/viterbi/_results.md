# Evaluation results

## English morphology
The dataset for this are the morph segmentations in [`bpe_knockout`](https://github.com/bauwenst/BPE-knockout)'s version 
of CELEX with the `legacy` flag turned *on*.

The baseline to beat is BPE-knockout, which does Pr=53%, Re=75%, F1=62% on morphs.

### BPE vocabulary
#### Probability-based
**CANINE with boundary-only symmetric probabilities** (2*P-1) and English BPE vocabulary as constraint:
```
BoMMa(BoundaryScoresChosen(LinearPT(-1,+1)) + VocabularyConstraintExact)
    Precision: 0.5707395498392283
    Recall:    0.8034367141659682
    F1:        0.6673861579167247
```
which is respectively +4%, +5%, +4% on BPE-knockout.

The vocabulary does matter a lot. The character model by itself does 92% on all metrics, and if you switch to a Dutch
vocabulary (RobBERT's) with the same objective, you get
```
BoMMa(BoundaryScoresChosen(LinearPT(-1,+1)) + VocabularyConstraintExact)
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
BoMMa(BoundaryScoresChosen(LogPT) + VocabularyConstraintExact)
    Precision: 0.5128729963008631
    Recall:    0.5810841017043867
    F1:        0.5448519779931884
    WW-Precision: 0.149173859432799
    WW-Recall:    0.8756514186450493
    WW-F1:        0.2549201399131864
```

Using joint log probabilities, which is probabilistically the most sound, actually produces the best tokeniser so far!
```
BoMMa(BoundaryScoresAll(LogPT) + VocabularyConstraintExact)
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
BoMMa(BoundaryScoresChosen(LinearPT(-1,+1)) + VocabularyConstraintExact)
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
BoMMa(BoundaryScoresAll(LogPT) + VocabularyConstraintExact)
    Precision: 0.5824413277045277
    Recall:    0.8133836267113719
    F1:        0.6788075223560411
```

And again, scores are worst for boundary-only log probabilities:
```
BoMMa(BoundaryScoresChosen(LogPT) + VocabularyConstraintExact)
    Precision: 0.5452460090819718
    Recall:    0.6508521933500978
    F1:        0.5933869981658856
```

Using the **AtLeastAll constraint with symmetric probability objective** unsurprisingly gives a result that is
at least as good as with an exact vocabulary:
```
BoMMa(BoundaryScoresAll(LogPT) + VocabularyConstraintAtLeastAll)
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

BoMMa(BoundaryScoresChosen(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

BoMMa(BoundaryScoresAll(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5839858651568084
    Recall:    0.8126851075719475
    F1:        0.6796111967848965  ---> NEW BEST!

---------------------------------------
YES EQUIVALENCE (which implies that with this objective, you can't actually take joint effect into account):
Also equivalent to the first LinearPT above.

BoMMa(BoundaryScoresChosen(LinearPT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

BoMMa(BoundaryScoresAll(LinearPT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.586417157275021
    Recall:    0.7792679519418833
    F1:        0.6692261547690462

---------------------------------------
NO EQUIVALENCE:

BoMMa(BoundaryScoresChosen(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5861179889091244
    Recall:    0.8003073484213468
    F1:        0.6766675722604802

BoMMa(BoundaryScoresAll(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Precision: 0.5839858651568084
    Recall:    0.8126851075719475
    F1:        0.6796111967848965

---------------------------------------
YES EQUIVALENCE:
(Again equivalent to the first class above.)

BoMMa(BoundaryScoresChosen(PiecewisePT(-2,+1)_NegComp) + VocabularyConstraintExact)
    Precision: 0.5861179889091244
    Recall:    0.8003073484213468
    F1:        0.6766675722604802

BoMMa(BoundaryScoresAll(PiecewisePT(-2,+1)_NegComp) + VocabularyConstraintExact)
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

Also notice the only two real BoundaryScoresAll results are *the same* regardless of LinearPT or PiecewisePT. What's more:
they are *equal across punishments*.
```
BoMMa(BoundaryScoresAll(LinearPT(-0.25,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084
    Re: 0.8126851075719475
    F1: 0.6796111967848965

BoMMa(BoundaryScoresAll(PiecewisePT(-0.25,+1)) + VocabularyConstraintExact):
    Pr: 0.5839858651568084
    Re: 0.8126851075719475
    F1: 0.6796111967848965
  
BoMMa(BoundaryScoresAll(LinearPT(-0.33,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084
    Re: 0.8126851075719475
    F1: 0.6796111967848965
  
BoMMa(BoundaryScoresAll(PiecewisePT(-0.33,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084
    Re: 0.8126851075719475
    F1: 0.6796111967848965

BoMMa(BoundaryScoresAll(LinearPT(-0.5,+1)) + VocabularyConstraintExact)
      Pr: 0.5839858651568084
      Re: 0.8126851075719475
      F1: 0.6796111967848965

BoMMa(BoundaryScoresAll(PiecewisePT(-0.5,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,
  
BoMMa(BoundaryScoresAll(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,

BoMMa(BoundaryScoresAll(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,

BoMMa(BoundaryScoresAll(LinearPT(-3,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,

BoMMa(BoundaryScoresAll(PiecewisePT(-3,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,

BoMMa(BoundaryScoresAll(LinearPT(-4,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965,

BoMMa(BoundaryScoresAll(PiecewisePT(-4,+1)) + VocabularyConstraintExact)
    Pr: 0.5839858651568084,
    Re: 0.8126851075719475,
    F1: 0.6796111967848965
```
Weird, and I guess we can never be sure if it isn't a bug (since we've had so many even after evaluation), but the code
is virtually the same as BoundaryScoresChosen with the addition of 2 lines of code...

BoundaryScoresChosen *does* have variable results, with best F1 so far for exact constraints:
```
BoMMa(BoundaryScoresChosen(LinearPT(-0.25,+1)) + VocabularyConstraintExact)
    Pr: 0.5384182846558396,
    Re: 0.909639564124057,
    F1: 0.6764461436170213
  
BoMMa(BoundaryScoresChosen(LinearPT(-0.33,+1)) + VocabularyConstraintExact)
    Pr: 0.5491143867320273,
    Re: 0.8982676725342275,
    F1: 0.6815777478613906,
  
BoMMa(BoundaryScoresChosen(LinearPT(-0.5,+1)) + VocabularyConstraintExact)
    Pr: 0.5645647815327927,
    Re: 0.8733165688739871,
    F1: 0.6857920200103124,  ---> New best, but precision is 2% lower than before.
  
BoMMa(BoundaryScoresChosen(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.586417157275021,  ---> New best, at the cost of recall
    Re: 0.7792679519418833,
    F1: 0.6692261547690462,
  
BoMMa(BoundaryScoresChosen(LinearPT(-3,+1)) + VocabularyConstraintExact)
    Pr: 0.585579629787597,
    Re: 0.7672254819782062,
    F1: 0.6642074453931932,
  
BoMMa(BoundaryScoresChosen(LinearPT(-4,+1)) + VocabularyConstraintExact)
    Pr: 0.5842290046740044,
    Re: 0.7578653255099189,
    F1: 0.6598148801342789,
```
Notice how the recall drops monotonously and the precision rises up to a certain point and then teeters down.

Same story for piecewise, although less prominently.
```
BoMMa(BoundaryScoresChosen(PiecewisePT(-0.25,+1)) + VocabularyConstraintExact)
    Pr: 0.5641620821377004,
    Re: 0.8624476110645432,
    F1: 0.6821210346618344,
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-0.33,+1)) + VocabularyConstraintExact)
    Pr: 0.5672451994091581,
    Re: 0.8583962000558816,
    F1: 0.6830906058921623,
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-0.5,+1)) + VocabularyConstraintExact)
    Pr: 0.5733971871509966,
    Re: 0.8463816708577815,
    F1: 0.683645719315271,
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.5861179889091244,
    Re: 0.8003073484213468,
    F1: 0.6766675722604802,
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-3,+1)) + VocabularyConstraintExact)
    Pr: 0.586262046339963,
    Re: 0.7988823693769209,
    F1: 0.6762535477767265,
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-4,+1)) + VocabularyConstraintExact)
    Pr: 0.5863377998481522,
    Re: 0.7983794355965353,
    F1: 0.676123658649125,
```
It's interesting that precision seems to stagnate.

You could interpret an increase in punishment as a push towards least-token Viterbi, with a tiebreaker on morphological 
matches (i.e. you find the least amount of tokens to use, and then try to find the path with that amount of tokens that has the
most good splits). I could see that explaining how you get from 51% precision and 58% recall to 59% precision (not that
much because you only have, say, 1 less bad split in most words) and 80% recall (the 1 extra good split you now prefer
makes up a significant amount of all good splits).

Apparently it's not the whole story though, since a literal implementation of least-token with boundary probability as
tiebreaker gives only 55% precision (+4% not +8%) and 64% recall (+6% not +16%). (See below.)

If true, that kinda sucks, because it means you can't increase precision arbitrarily. It makes sense though, because not
all words can be represented by 1 token, so you just can't keep asking for fewer tokens and expect the tokeniser to need
fewer steps. Also, I'd like to remind that it's not obvious that you would *want* to push towards using fewer and fewer tokens.
The subword vocabulary contains many full words that contain more than one morpheme (e.g. words with -ion and so on).
Even though the tokeniser *can* jump to the end immediately, it shouldn't, perhaps even if it costs 1 bad split to get
1 good split out of it.

The best punishment seems to lie between -0.5 and -2 for both the linear and the piecewise transform.
To repeat those results, ordered in increasing precision (and coincidentally decreasing recall!):
```
BoMMa(BoundaryScoresChosen(LinearPT(-0.5,+1)) + VocabularyConstraintExact)
    Pr: 0.5645647815327927
    Re: 0.8733165688739871  
    F1: 0.6857920200103124  ---> New best, but precision is 2% lower than before.
    
BoMMa(BoundaryScoresChosen(PiecewisePT(-0.5,+1)) + VocabularyConstraintExact)
    Pr: 0.5733971871509966
    Re: 0.8463816708577815
    F1: 0.683645719315271

BoMMa(BoundaryScoresChosen(LinearPT/PiecewisePT(-1,+1)) + VocabularyConstraintExact)
    Pr: 0.583957433992571
    Re: 0.8126292260407936
    F1: 0.6795724049301947
  
BoMMa(BoundaryScoresChosen(PiecewisePT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.5861179889091244
    Re: 0.8003073484213468
    F1: 0.6766675722604802
          
BoMMa(BoundaryScoresChosen(LinearPT(-2,+1)) + VocabularyConstraintExact)
    Pr: 0.586417157275021  ---> New best, at the cost of recall
    Re: 0.7792679519418833
    F1: 0.6692261547690462
```
Also interesting how -1/+1, the centre of these, is approximated so well by `BoundaryScoresAll(LogPT) + Exact`.

#### Hard-boundary-based
One way to disincentivise splitting, as seen above, is to give a low score to non-boundaries in the same grid you score
boundaries highly.

A different approach is to optimise for probability and, of the paths with the same maximal boundary probability (among
which is the full character segmentation, which hits every boundary), pick the one with fewest tokens. For the first step
you need to discretise your probabilities so that small probabilities on negative boundaries don't give the
character segmentation an edge (incentivising segmentation).

As expected, you get high recall (although you would really expect almost 100%...), but
no fundamental improvements over the disincentivisation using negative scores rather than tiebreaking the best positive scorers.
```
ProbabilityViterbiWithLeastTokenTiebreaker
    Precision: 0.5450523702110546
    Recall:    0.8746071829405163
    F1:        0.6715789246894828
```

#### Hard-boundary-prefix-based
Prefix-based objectives give a higher score when they include a larger portion of the start of a gold token.

For exact vocabulary constraint, we are looking to beat 58%, 81%, 68%:
```
BoMMa(BoundaryPrefixLength(pm=0) + VocabularyConstraintExact)
    Precision: 0.5290765557743583
    Recall:    0.8482816429170159
    F1:        0.6516909405085165

BoMMa(BoundaryPrefixLength(pm=-1) + VocabularyConstraintExact)
    Precision: 0.5359700205844579
    Recall:    0.8511874825370215
    F1:        0.6577637672867028

BoMMa(BoundaryPrefixLengthExtended(pm=0) + VocabularyConstraintExact)
    Precision: 0.5411118078866797
    Recall:    0.8197261804973456
    F1:        0.6518976091014131

BoMMa(BoundaryPrefixLengthExtended(pm=-1) + VocabularyConstraintExact)
    Precision: 0.5541110599785243
    Recall:    0.8074322436434759
    F1:        0.6572058856973915
```
The difference between the non-extended and extended version is that the extended ones have higher
precision, likely because they make fewer splits since they get rewarded for overshooting boundaries as
long as the step starts on a boundary. It also makes sense that their recall is lowest because they don't take
small steps in order to still catch some reward they would've lost by overshooting.
TL;DR: giving credit even when you overshoot lowers the amount of splits and hence ups precision while lowering recall.

Punishment seems to increase precision but it's unclear what it does to recall.
The higher precision makes sense since punishment disincentivises oversegmentation. 
We should vary the punishments though.

```
BoMMa_Sum(BoundaryPrefixLength(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5290765557743583
   Recall:    0.8482816429170159
   F1:        0.6516909405085165

BoMMa_Sum(BoundaryPrefixLength_Normed(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5290765557743583
   Recall:    0.8482816429170159
   F1:        0.6516909405085165

BoMMa_Sum(BoundaryPrefixLength(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5359700205844579
   Recall:    0.8511874825370215
   F1:        0.6577637672867028

BoMMa_Sum(BoundaryPrefixLength_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5432766615146831
   Recall:    0.8446214026264319
   F1:        0.6612345787032986

BoMMa_Sum(BoundaryPrefixLength(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5392776043145047
   Recall:    0.849343392008941
   F1:        0.6596931357017296

BoMMa_Sum(BoundaryPrefixLength_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5524127691165553
   Recall:    0.8316289466331378
   F1:        0.6638563622170179

BoMMa_Sum(BoundaryPrefixLengthExtended(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5411118078866797
   Recall:    0.8197261804973456
   F1:        0.6518976091014131

BoMMa_Sum(BoundaryPrefixLengthExtended_Normed(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5292527683320255
   Recall:    0.8480022352612462
   F1:        0.6517420948086111

BoMMa_Sum(BoundaryPrefixLengthExtended(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5541110599785243
   Recall:    0.8074322436434759
   F1:        0.6572058856973915

BoMMa_Sum(BoundaryPrefixLengthExtended_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5669964209595045
   Recall:    0.7879016485051691
   F1:        0.6594406248538422

BoMMa_Sum(BoundaryPrefixLengthExtended(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5617480293182132
   Recall:    0.7944677284157586
   F1:        0.658141122825697

BoMMa_Sum(BoundaryPrefixLengthExtended_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5697980684811238
   Recall:    0.7797429449566918
   F1:        0.6584401948871613

BoMMa_Sum(BoundarySuffixLength(pm=0)) + VocabularyConstraintExact)
   Precision: 0.527421487012755
   Recall:    0.8249231628946633
   F1:        0.6434486967134513

BoMMa_Sum(BoundarySuffixLength_Normed(pm=0)) + VocabularyConstraintExact)
   Precision: 0.527421487012755
   Recall:    0.8249231628946633
   F1:        0.6434486967134513

BoMMa_Sum(BoundarySuffixLength(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5323896094564007
   Recall:    0.8280525286392847
   F1:        0.6480930720783764

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5544226498889712
   Recall:    0.8371332774518022
   F1:        0.6670600022264277

BoMMa_Sum(BoundarySuffixLength(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5472073618287717
   Recall:    0.837384744341995
   F1:        0.6618889342859351

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5628878577838881
   Recall:    0.8236658284436994
   F1:        0.6687537573303389

BoMMa_Sum(BoundarySuffixLengthExtended(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5288189089343998
   Recall:    0.8239172953338921
   F1:        0.6441803565186998

BoMMa_Sum(BoundarySuffixLengthExtended_Normed(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5280878774867611
   Recall:    0.8247555183012014
   F1:        0.6438932879610846

BoMMa_Sum(BoundarySuffixLengthExtended(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.539861479516652
   Recall:    0.8188879575300363
   F1:        0.6507249272852418

BoMMa_Sum(BoundarySuffixLengthExtended_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5778501493528045
   Recall:    0.778345906677843
   F1:        0.663277697088026

BoMMa_Sum(BoundarySuffixLengthExtended(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5631739493675805
   Recall:    0.8098910310142498
   F1:        0.6643669993926128

BoMMa_Sum(BoundarySuffixLengthExtended_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5801030602587023
   Recall:    0.7706342553785974
   F1:        0.6619307613847724

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended(pm=0)) + VocabularyConstraintExact)
   Precision: 0.5523155130998268
   Recall:    0.8287510477787091
   F1:        0.6628674868425464

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended_Normed(pm=0)) + VocabularyConstraintExact)
   Precision: 0.545613037015287
   Recall:    0.8307069013690975
   F1:        0.6586324918864435

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5533579859554759
   Recall:    0.8278569432802458
   F1:        0.6633308706651444

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5518945634266886
   Recall:    0.8236937692092763
   F1:        0.6609420890971459

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5554428668018931
   Recall:    0.8263202011735121
   F1:        0.6643304804905992

BoMMa_Sum(BoundaryPrefixAndSuffixLengthExtended_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5521809188338455
   Recall:    0.8234423023190836
   F1:        0.6610663735672149
```
Some conclusions:
- Norming seems to produce better F1.
- Prefix objectives only attain >66% F1 on two occasions, whilst suffix objectives do it consistently.
- More punishment seems to produce better F1.
- Norming does nothing when there is no punishment and no extension.
- Overall, `BoundarySuffixLength_Normed` with high punishment seems to do best.

Pulling up the punishment on `BoundarySuffixLength_Normed` has effect until about -3.5 but
is best at -2.
```
BoMMa_Sum(BoundarySuffixLength_Normed(pm=-0.5)) + VocabularyConstraintExact)
   Precision: 0.5519227946572729
   Recall:    0.8405141100866164
   F1:        0.6663122688107738

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-1)) + VocabularyConstraintExact)
   Precision: 0.5544226498889712
   Recall:    0.8371332774518022
   F1:        0.6670600022264277

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-1.5)) + VocabularyConstraintExact)
   Precision: 0.5624106308745305
   Recall:    0.8242246437552389
   F1:        0.6686007638172731

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-2)) + VocabularyConstraintExact)
   Precision: 0.5628878577838881
   Recall:    0.8236658284436994
   F1:        0.6687537573303389

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-2.5)) + VocabularyConstraintExact)
   Precision: 0.5638342804484718
   Recall:    0.8205923442302319
   F1:        0.6684039236213841

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-3)) + VocabularyConstraintExact)
   Precision: 0.5639784016448569
   Recall:    0.8200614696842693
   F1:        0.668328949915178

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-3.5)) + VocabularyConstraintExact)
   Precision: 0.564188344773221
   Recall:    0.8198938250908074
   F1:        0.668420633022403

BoMMa_Sum(BoundarySuffixLength_Normed(pm=-4)) + VocabularyConstraintExact)
   Precision: 0.564188344773221
   Recall:    0.8198938250908074
   F1:        0.668420633022403
```

For at-least constraint, we are looking to beat 66%, 85%, 74%:
```
BoMMa(BoundaryPrefixLength(pm=0) + VocabularyConstraintAtLeastAll)
    Precision: 0.6116584912043301
    Recall:    0.8082984073763622
    F1:        0.6963628048046602

BoMMa(BoundaryPrefixLengthExtended(pm=0) + VocabularyConstraintAtLeastAll)
    Precision: 0.6116584912043301
    Recall:    0.8082984073763622
    F1:        0.6963628048046602

BoMMa(BoundaryPrefixLength(pm=-1) + VocabularyConstraintAtLeastAll)
    Precision: 0.6199262778854964
    Recall:    0.8129365744621403
    F1:        0.7034319354955697

BoMMa(BoundaryPrefixLengthExtended(pm=-1) + VocabularyConstraintAtLeastAll)
    Precision: 0.620031118784236
    Recall:    0.8127968706342554
    F1:        0.7034471084672398
```

#### Multiplicative
Rather than summing symmetric probabilities, you can also multiply raw probabilities. The issue with this is that if
you don't take into account the unchosen boundaries (if you do, then you get `BoundaryScoresAll` with summing and `LogPT`),
you disincentivise making more splits by default since multiplying by nothing is better than 0.9999.

We have come up with transformations that fix this. The power transform `1 + c (x-0.5)` slash `1/(1 + c (0.5 - x))` 
seems to do really well! Precision is consistently 58.7% which is higher than anything we've seen, and F1 is highly
competitive.
```
MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=0.25,p=+1)) + VocabularyConstraintExact)
    Precision: 0.587749218337597
    Recall:    0.8122615039281706
    F1:        0.6820034395834806

MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=0.5,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5877086124984773
    Recall:    0.8122053872053872
    F1:        0.681956322001555

MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=0.75,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5876179843702426
    Recall:    0.8122615039281706
    F1:        0.6819150815617455

MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=1.0,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5875512440638064
    Recall:    0.812317620650954
    F1:        0.681889912146407

MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=1.1,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5875489539579148
    Recall:    0.8124298540965208
    F1:        0.6819279094688004

MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=1.25,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5875131888645402
    Recall:    0.8124298540965208
    F1:        0.6819038198860158
        
MultiplicativeBalanceViterbi(BoundaryScoresChosen(PowerMBPT(c=2,p=+1)) + VocabularyConstraintExact)
    Precision: 0.5873608209787657
    Recall:    0.812598204264871
    F1:        0.6818604541655385
```
With increasing `c`, i.e. a more extreme transform, precision drops a little and recall rises a little.

#### Unguided
Using **least-token** Viterbi with ULM vocabulary:
```
LeastTokenViterbi
    Precision: 0.5134891184434861
    Recall:    0.574635241301908
    F1:        0.5423441555002384
```

Adding a guided tiebreaker (`BoundaryScoresChosen` with no transform applied to the probabilities) is slightly better:
```
LeastTokenViterbiWithProbabilityTiebreaker
    Precision: 0.5498700730990601
    Recall:    0.635297418630752
    F1:        0.5895049272947395
```
... but it is much worse (-24% recall and -9% F1) than if you swap the main objective with the tiebreaker, see above.