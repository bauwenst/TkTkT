"""
Very reduced version of the train() function to fine-tune CANINE,
to load an already fine-tuned model and test it on the entire dataset.
"""
from tst.preamble import *

from tst.evaluation.english_morphology import make_CanineViterbiBPE

from tktkt.models.neural.canine_finetuning import datasetOutOfContext, dataloaderFromIterable, compute_metrics, MAX_INPUT_LENGTH_CANINE, tokenizer
from tktkt.models.viterbi.instances import HuggingFaceCharacterModelForTokenClassification, ConstrainVocabulary, ScoreGeneratorUsingCharacterClassifier

from datasets import concatenate_datasets
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

# Get model from another test script.
canine_viterbi = make_CanineViterbiBPE()
generator: ConstrainVocabulary = canine_viterbi.objectives[0].score_generator
nested_generator: ScoreGeneratorUsingCharacterClassifier = generator.nested_generator
classifier: HuggingFaceCharacterModelForTokenClassification = nested_generator.logprob_classifier
model = classifier.model
model.to("cuda")


##################################


def evaluate(only_testset=False):
    BATCH_SIZE = 32

    # Get dataset
    datasetdict = dataloaderFromIterable(datasetOutOfContext())
    collator = DataCollatorForTokenClassification(tokenizer, padding="longest", max_length=MAX_INPUT_LENGTH_CANINE)

    print(datasetdict)

    # Useless training arguments that will be used for nothing.
    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=BATCH_SIZE,

        # Artifacts
        report_to="none",   # Disables weights-and-biases login requirement
        logging_strategy="no",
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        data_collator=collator,
        # train_dataset=datasetdict["train"],  # Might crash, dunno
        eval_dataset=datasetdict["test"] if only_testset else concatenate_datasets([datasetdict["train"], datasetdict["valid"], datasetdict["test"]]),
        compute_metrics=compute_metrics
    )
    print(trainer.evaluate())


if __name__ == "__main__":
    evaluate(only_testset=False)
