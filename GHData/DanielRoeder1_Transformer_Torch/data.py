from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


class Wmt14Handler():
    def __init__(self, tokenizer,config, language_split) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.language_split = language_split
    
    def preprocess_function(self,examples):
        """
        Shifts the targets one token to the right using bos_token
        """
        lang1, lang2 = list(examples["translation"][0].keys())
        inputs = [ex[lang1] for ex in examples["translation"]]
        targets = [ex[lang2] for ex in examples["translation"]]

        model_inputs = self.tokenizer(
            inputs, text_target=targets, add_special_tokens= False, max_length= self.config.seq_len, truncation= True, return_attention_mask= False, return_token_type_ids= False, padding= "max_length"
        )
        model_inputs["labels"] = [ ids + [self.tokenizer.pad_token_id] for ids in model_inputs["labels"]]
        return model_inputs

    def get_wmt14(self):
        if self.config.local_data_path:
            tokenized_dataset = load_from_disk(self.config.local_data_path)
        else:
            dataset = load_dataset("wmt14", self.language_split)
            tokenized_dataset = dataset.map(self.preprocess_function, batched= True)
            tokenized_dataset = tokenized_dataset.remove_columns("translation")
            tokenized_dataset.save_to_disk("dataset.hf")
        tokenized_dataset = tokenized_dataset.with_format("torch")
        print("Loaded Data!")
        return tokenized_dataset
    
def get_Tokenizer():
    """
    We update the BERT WMT tokenizer to be used for our task.
    1. Introduce BOS and EOS token and update the postprocessing so they are added if add_special_tokens = True
    2. Change the unk token as it was misaligned with the one used in WordPiece leading to an error if an unknown word as found
    """
    tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", bos_token = "[BOS]", eos_token = "[EOS]", use_fast = True)
    tokenizer._tokenizer.model.unk_token = "<unk>"
    tokenizer._tokenizer.post_processor = TemplateProcessing(single= "[BOS] $0 [EOS]", special_tokens=[("[BOS]", tokenizer.bos_token_id), ("[EOS]", tokenizer.eos_token_id)])
    return tokenizer


if __name__ == "__main__":
    from utils import ConfigObject
    tokenizer = get_Tokenizer()
    config = ConfigObject("config.json")

    t = Wmt14Handler(tokenizer, config, "de-en")
    data = t.get_wmt14()


    
