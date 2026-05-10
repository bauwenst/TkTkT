from transformers import AutoTokenizer

from tktkt.models.linguistic.dele.training import DelTrainer
from tktkt.models.linguistic.dele.segmentation import EnglishDerivator
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser

bert_tkz = HuggingFaceTokeniser(AutoTokenizer.from_pretrained("bert-base-uncased"))

trainer = DelTrainer(EnglishDerivator, bert_tkz)
trainer.train_coarse(length_limit=4)
