# TODO: There are two ways to set up training.
#   - Out of context: you give only the word and its labels, and ask the model to predict at each character.
#   - In context: you use the pre-training corpus to get sentences that contain at least one word from the dataset.
#                 then you do the same as MLM where only a select few tokens are actually used for prediction, here
#                 the characters part of the word of interest. Much richer data. Would be loaded completely differently
#                 though (OSCAR but filtered by checking if any word is in the lemma set, and with labels constructed
#                 by padding the given labels with long sequences of -100).
import re

import torch
from datasets import Dataset, DatasetDict
from transformers import CanineForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from transformers.models.canine.modeling_canine import CanineForTokenClassification as CFTC
from transformers.models.canine.tokenization_canine import CanineTokenizer
import torch
import evaluate

from bpe_knockout.project.config import morphologyGenerator


TRAINING_EPOCHS = 3
BATCH_SIZE = 32
MAX_INPUT_LENGTH_CANINE = 2048


def datasetOutOfContext():
    print("> Building dataset")

    BAR = "|"
    FIND_BAR = re.compile(re.escape(BAR))
    for obj in morphologyGenerator():
        splitstring = obj.morphSplit()
        split_indices = [match.start() // 2 for match in FIND_BAR.finditer(" ".join(splitstring).replace("   ", BAR))]

        text = obj.lemma()
        labels = torch.zeros(len(text), dtype=torch.int8)
        labels[split_indices] = 1
        yield {"text": text, "labels": labels}


print("> Loading tokeniser")
tokenizer: CanineTokenizer = AutoTokenizer.from_pretrained("google/canine-s")  # There is no unk_token because any Unicode codepoint is mapped via a hash table to an ID (which is better than UTF-8 byte tokenisation although not reversible).


def dataloaderFromIterable(iterable):
    def preprocess(example):
        output = tokenizer(example["text"], add_special_tokens=False, # return_tensors="pt",  # DO NOT USE THIS OPTION, IT IS EVIL. Will basically make 1-example batches of everything even though things like the collator will expect non-batches, and hence they will think no padding is needed because all features magically have the same length of 1.
                           truncation=True, max_length=MAX_INPUT_LENGTH_CANINE)
        output["labels"] = example["labels"]
        return output

    dataset = Dataset.from_list(list(iterable))
    dataset = dataset.map(preprocess, batched=False)
    dataset = dataset.remove_columns(["text"])

    # 80-10-10 split
    datasetdict_train_vs_validtest = dataset.train_test_split(train_size=80/100)
    datasetdict_valid_vs_test      = datasetdict_train_vs_validtest["test"].train_test_split(train_size=50/100)
    return DatasetDict({
        "train": datasetdict_train_vs_validtest["train"],
        "valid": datasetdict_valid_vs_test["train"],
        "test": datasetdict_valid_vs_test["test"]
    })


def train():
    # Set up paths for checkpointing  TODO: If you do it relative to the root, this only works in an editable install.
    from src.tktkt.files.paths import setTkTkToutputRoot, PATH_ROOT, getTkTkToutputPath
    setTkTkToutputRoot(PATH_ROOT / "data" / "out")
    output_goes_under_here = getTkTkToutputPath()  # runs a mkdir
    PATH_CHECKPOINTS = output_goes_under_here / "checkpoints"
    PATH_CHECKPOINTS.mkdir(exist_ok=True)

    # Get dataset
    datasetdict = dataloaderFromIterable(datasetOutOfContext())

    # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
    # Note that unlike sequence classification, the labels per example have variable length in token classification.
    # Since you pad input IDs, you must also pad labels. Otherwise, you will get the very confusing
    #       ValueError: Unable to create tensor, you should probably activate truncation and/or padding with
    #       'padding=True' 'truncation=True' to have batched tensors with the same length.
    collator = DataCollatorForTokenClassification(tokenizer, padding="longest", max_length=MAX_INPUT_LENGTH_CANINE)

    # Get model
    model: CFTC = CanineForTokenClassification.from_pretrained("google/canine-s", num_labels=20)
    model.to("cuda")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=PATH_CHECKPOINTS.as_posix(),
        save_strategy="no",
        # load_best_model_at_end=False,

        num_train_epochs=TRAINING_EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,  # L2 regularisation constant
        report_to="none",   # Disables weights-and-biases login requirement

        evaluation_strategy="epoch",
        logging_strategy="no"
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        data_collator=collator,
        train_dataset=datasetdict["train"],

        eval_dataset=datasetdict["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    print(trainer.evaluate())


def compute_metrics(stuff):
    print(stuff)



# Other examples
##################################################

def exampleInference():
    print("> Loading model")
    model: CFTC = CanineForTokenClassification.from_pretrained("google/canine-s", num_labels=20)

    print("> Tokenising")
    inputs = tokenizer("HuggingFace is a company based in Paris and New York",
                       add_special_tokens=False, return_tensors="pt")  # Note: if you do add special tokens (which you could motivate based on having "working memory"), you likely need to surround the labels tensor by -100.

    print("> Inferring")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_token_class_ids = logits.argmax(-1)

    # Note that tokens are classified rather then input words which means that
    # there might be more predicted token classes than words.
    # Multiple token classes might account for the same word
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    print(predicted_tokens_classes)


def exampleMetrics():
    metric = evaluate.combine([
        evaluate.load("accuracy"),
        evaluate.load("precision"),
        evaluate.load("recall"),
        evaluate.load("f1")
    ])
    print(metric.compute(predictions=[0,1,1,1,1,1], references=[0,0,0,0,1,0]))


def exampleDataset():
    for thing in dataloaderFromIterable(datasetOutOfContext())["train"]:
        print(thing)


if __name__ == "__main__":
    # exampleDataset()
    train()