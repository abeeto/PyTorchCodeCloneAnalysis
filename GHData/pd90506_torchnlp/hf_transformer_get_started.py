# %%
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
# %%
classifier('We are very happy to show you the Transformers library.')
# %%
results = classifier(["We are very happy to show you the Transformers\
    library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# %%
