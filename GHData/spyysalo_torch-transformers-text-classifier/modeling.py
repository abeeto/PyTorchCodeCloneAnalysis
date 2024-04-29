# Text classification using models not supported by
# AutoModelForSequenceClassification

import copy

import torch

from torch import nn

from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import SequenceClassifierOutput


def mean_pool(token_embeddings, attention_mask):
    # Following https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L98
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


class T5ForSequenceClassification(T5PreTrainedModel):
    # Implementation adapting T5ForConditionalGeneration and
    # BertForSequenceClassification
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        hidden_states = encoder_outputs[0]

        pooled_output = mean_pool(hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is None:
            loss = None
        else:
            if self.config.problem_type is None:
                if labels.dtype == torch.long or labels.dtype == torch.int:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
            )
