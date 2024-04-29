# %%
from transformers import DistilBertModel
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
# Initializing a DistilBERT configuration
configuration = DistilBertConfig()
model = DistilBertModel(configuration)
configuration = model.config
# %%
