from transformers import BertModel, BertTokenizer

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<|unkwn|>")
    tokenizer.save_vocabulary('bert')
    model = BertModel.from_pretrained("bert-base-uncased")
    model.save_pretrained('bert')

if __name__ == "__main__":
    main()