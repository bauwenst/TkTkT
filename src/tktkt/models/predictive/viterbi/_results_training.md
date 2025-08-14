# Evaluation results
## English morphology
I fine-tuned CANINE-C on 80% of the CELEX-en morphologies, stopping when loss on a 10% validation split converges with
a patience of half an epoch. 
Using the argmax of its predicted probability of split vs. non-split as decision, you get:

Test set only:
Precision: 0.9253731343283582
Recall:    0.9268823049741778
F1:        0.926127104834329

Entire dataset (train and validation set too):
Precision: 0.9219220895189195
Recall:    0.9231070131321598
F1:        0.9225141708318209

Kind of spectacular that it performs better on the data it has never seen than the data that were used to inform its
training. 

Note that this isn't a practical subword tokeniser since it assumes an infinite subword vocabulary.

### Improved stopping condition
By increasing the patience to 3 full epochs and also changing the convergence criterion from loss to evaluation recall,
longer training and even better results were achieved:
```
Test set:
    Precision: 0.945273018612104
    Recall:    0.9493686556078237
    F1:        0.9473164103514298
```