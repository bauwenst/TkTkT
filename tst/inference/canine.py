import torch

from transformers import CanineForTokenClassification
from transformers.models.canine.modeling_canine import CanineForTokenClassification as CFTC
from transformers.models.canine.tokenization_canine import CanineTokenizer


def exampleInference():
    print("> Loading model")
    model: CFTC = CanineForTokenClassification.from_pretrained("google/canine-c", num_labels=20)

    print("> Tokenising")
    tokenizer: CanineTokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    inputs = tokenizer("HuggingFace is a company based in Paris and New York",
                       add_special_tokens=False, return_tensors="pt")  # Note: if you do add special tokens (which you could motivate based on having "working memory"), you likely need to surround the labels tensor by -100.

    print("> Inferring")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_token_class_ids = logits.argmax(-1)
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    print(predicted_tokens_classes)
