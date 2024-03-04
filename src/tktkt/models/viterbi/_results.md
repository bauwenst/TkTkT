# Evaluation results

## English morphology
CANINE with boundary-only symmetric probabilities (2*P-1) performs as follows:
		Precision: 0.20260564287554753
		Recall:    0.46133255106156723
		F1:        0.2815580695334967
That can't be right... Something buggy is happening. There is too much intelligence in this model for it to do so poorly
given that BPE does it much better and is dumb.

Yup. The bug is that although the Viterbi is doing everything properly, the pretokeniser undo() is stripping characters off of tokens.
It's AppendSpace that's doing this.
We predicted that this would happen, because this was originally applied to pretokens (append 1 space to a pretoken that
will be split in the future) whilst now it is applied to EVERY token. Indeed, there are pretokenisation steps that
should only be applied after re-concatenating the tokens (except you don't know how much to concatenate them... messy!).

After a quick fix, it's now
    Precision: 0.43320889909887733
    Recall:    0.8851913942442023
    F1:        0.5817243690381102
which is respectively -10%, +13% and -4% over English BPE-knockout (53%, 75%, 62%). Clearly oversegmenting...
Aha, but we were using RobBERT's vocabulary for Viterbi steps, which is a Dutch vocabulary, not English!
With English BPE vocabulary:
    Precision: 0.5707395498392283
    Recall:    0.8034367141659682
    F1:        0.6673861579167247
which is respectively +4%, +5%, +4% on BPE-knockout.

We have three model alternatives:
    - Symmetric probabilities but using the joint (i.e. also counting the non-boundaries);
    - Boundary-only log probabilities
    - Joint log probabilities
Strangely, the first one changes 0 of the predicted splits.

When you use log probabilities WITHOUT the joint, performance tanks.
    Precision: 0.5128729963008631
    Recall:    0.5810841017043867
    F1:        0.5448519779931884
    WW-Precision: 0.149173859432799
    WW-Recall:    0.8756514186450493
    WW-F1:        0.2549201399131864

Using joint log probabilities, which is probabilistically the most sound, actually produces the best tokeniser so far!
    Precision: 0.5703759458067583
    Recall:    0.8045822855546242
    F1:        0.6675321062636191
...with whole-word metrics beating BPE-knockout (12%, 89%, 21%) everywhere too:
    WW-Precision: 0.12900606108624174
    WW-Recall:    0.9428199189345686
    WW-F1:        0.22695752169216296

Token-minimising Viterbi performs markedly worse, with (no surprise) way worse recall:
    Precision: 0.4559803399964531
    Recall:    0.5028778988544286
    F1:        0.47828224445595985
In fact, its recall is so low that despite trying to find the biggest units possible, it underperforms BPE-knockout
in whole-word boundary recall by 5%:
    WW-Precision: 0.1464619594132401
    WW-Recall:    0.8368558193398957
    WW-F1:        0.249293861445913


Using symmetric probability (joint or not) and a ULM vocabulary:
    Precision: 0.583957433992571
    Recall:    0.8126292260407936
    F1:        0.6795724049301947

    WW-Precision: 0.13169360505973296
    WW-Recall:    0.9494788650839606
    WW-F1:        0.2313049918008217

...which is even better than anything BPE had, confirming that the vocab is the bottleneck.

The joint log probability has a lower precision and F1 this time:
    Precision: 0.5824413277045277
    Recall:    0.8133836267113719
    F1:        0.6788075223560411

And again, scores are worst for boundary-only log probabilities:
    Precision: 0.5452460090819718
    Recall:    0.6508521933500978
    F1:        0.5933869981658856
	
