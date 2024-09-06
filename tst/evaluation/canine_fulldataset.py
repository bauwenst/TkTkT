"""
Very reduced version of the train() function to fine-tune CANINE,
to load an already fine-tuned model and test it on the entire dataset.

FIXME: This entire file should really be replaced by an evaluation-only LaMoTO script.
       We'll need a special flag in LaMoTO that can set the eval dataset to all three splits.
           eval_dataset=datasetdict["test"] if only_testset else concatenate_datasets([datasetdict["train"], datasetdict["valid"], datasetdict["test"]])
       It would actually make some sense to have a .evaluate() method in LaMoTO tasks.
"""
from tst.preamble import *

from lamoto.trainer.hyperparameters import *
from lamoto.tasks.mbr import MBR, SUGGESTED_HYPERPARAMETERS_MBR
from tktkt.models.viterbi.instances import HuggingFaceCharacterModelForTokenClassification, VocabularyConstraintExact, ScoreGeneratorUsingCharacterClassifier
from tktkt.builders.english import Builder_English_CanineViterbi_BPE

# Model setup  TODO: I wonder how we can support loading completely custom models in LaMoTO.
canine_viterbi = Builder_English_CanineViterbi_BPE().buildTokeniser()
generator: VocabularyConstraintExact = canine_viterbi.objectives[0].score_generator
nested_generator: ScoreGeneratorUsingCharacterClassifier = generator.nested_generator
classifier: HuggingFaceCharacterModelForTokenClassification = nested_generator.logprob_classifier
model = classifier.model
model.to("cuda")

# Hyperparameters
hp = SUGGESTED_HYPERPARAMETERS_MBR
hp.HARD_STOPPING_CONDITION = AfterNEpochs(epochs=0, effective_batch_size=SUGGESTED_HYPERPARAMETERS_MBR.EXAMPLES_PER_EFFECTIVE_BATCH)

# "Train", which is just an evaluation.
mbr = MBR(dataset_out_of_context=True)
mbr.train()
