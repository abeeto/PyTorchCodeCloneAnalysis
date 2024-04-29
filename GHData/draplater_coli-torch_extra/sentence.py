import torch
from dataclasses import field, dataclass
from torch.nn import Module, ModuleDict, Embedding, LayerNorm

from coli.basic_tools.dataclass_argparse import argfield, OptionsBase
from coli.torch_extra.layers import CharacterEmbedding
from coli.torch_extra.dropout import FeatureDropout2


class SentenceEmbeddings(Module):
    @dataclass
    class Options(OptionsBase):
        dim_word: "word embedding dim" = 100
        dim_postag: "postag embedding dim. 0 for not using postag" = 100
        dim_char_input: "character embedding input dim" = 100
        dim_char: "character embedding dim. 0 for not using character" = 100
        word_dropout: "word embedding dropout" = 0.4
        postag_dropout: "postag embedding dropout" = 0.2
        character_embedding: CharacterEmbedding.Options = field(
            default_factory=CharacterEmbedding.Options)
        input_layer_norm: "Use layer norm on input embeddings" = True
        mode: str = argfield("concat", choices=["add", "concat"])
        replace_unk_with_chars: bool = False

    def __init__(self,
                 hparams: "SentenceEmbeddings.Options",
                 statistics,
                 plugins=None
                 ):

        super().__init__()
        self.hparams = hparams
        self.mode = hparams.mode
        self.plugins = ModuleDict(plugins) if plugins is not None else {}

        # embedding
        input_dims = {}
        if hparams.dim_word != 0:
            self.word_embeddings = Embedding(
                len(statistics.words), hparams.dim_word, padding_idx=0)
            self.word_dropout = FeatureDropout2(hparams.word_dropout)
            input_dims["word"] = hparams.dim_word

        if hparams.dim_postag != 0:
            self.pos_embeddings = Embedding(
                len(statistics.postags), hparams.dim_postag, padding_idx=0)
            self.pos_dropout = FeatureDropout2(hparams.postag_dropout)
            input_dims["postag"] = hparams.dim_postag

        if hparams.dim_char > 0:
            self.bilm = None
            self.character_lookup = Embedding(len(statistics.characters),
                                              hparams.dim_char_input)
            self.char_embeded = CharacterEmbedding.get(hparams.character_embedding,
                                                       dim_char_input=hparams.dim_char_input,
                                                       input_size=hparams.dim_char)
            if not hparams.replace_unk_with_chars:
                input_dims["char"] = hparams.dim_char
            else:
                assert hparams.dim_word == hparams.dim_char
        else:
            self.character_lookup = None

        for name, plugin in self.plugins.items():
            input_dims[name] = plugin.output_dim

        if hparams.mode == "concat":
            self.output_dim = sum(input_dims.values())
        else:
            assert hparams.mode == "add"
            uniq_input_dims = list(set(input_dims.values()))
            if len(uniq_input_dims) != 1:
                raise ValueError(f"Different input dims: {input_dims}")
            print(input_dims)
            self.output_dim = uniq_input_dims[0]

        self.input_layer_norm = LayerNorm(self.output_dim, eps=1e-6) \
            if hparams.input_layer_norm else None

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.word_embeddings.weight.data)
        if self.hparams.dim_postag != 0:
            torch.nn.init.xavier_normal_(self.pos_embeddings.weight.data)
        if self.character_lookup is not None:
            torch.nn.init.xavier_normal_(self.character_lookup.weight.data)

    def forward(self, inputs, unk_idx=1):
        all_features = []

        if self.character_lookup is not None:
            # use character embedding instead
            # batch_size, bucket_size, word_length, embedding_dims
            char_embeded_4d = self.character_lookup(inputs.chars)
            word_embeded_by_char = self.char_embeded(inputs.word_lengths,
                                                     char_embeded_4d)
            if not self.hparams.replace_unk_with_chars:
                all_features.append(word_embeded_by_char)

        if self.hparams.dim_word != 0:
            word_embedding = self.word_dropout(self.word_embeddings(inputs.words))
            if self.hparams.dim_char and self.hparams.replace_unk_with_chars:
                unk = inputs.words.eq(unk_idx)
                # noinspection PyUnboundLocalVariable
                unk_word_embeded_by_char = word_embeded_by_char[unk]
                word_embedding[unk] = unk_word_embeded_by_char
            all_features.append(word_embedding)

        if self.hparams.dim_postag != 0:
            all_features.append(self.pos_dropout(self.pos_embeddings(inputs.postags)))

        for plugin in self.plugins.values():
            plugin_output = plugin(inputs)
            # FIXME: remove these two ugly tweak
            if plugin_output.shape[1] == inputs.words.shape[1] + 2:
                plugin_output = plugin_output[:, 1:-1]
            # pad external embedding to dim_word
            # if self.mode == "add" and plugin_output.shape[-1] < self.hparams.dim_word:
            #     plugin_output = torch.cat(
            #         [plugin_output,
            #          plugin_output.new_zeros(
            #              (*inputs.words.shape, self.hparams.dim_word - plugin_output.shape[-1]))], -1)
            all_features.append(plugin_output)

        if self.mode == "concat":
            total_input_embeded = torch.cat(all_features, -1)
        else:
            total_input_embeded = sum(all_features)

        if self.input_layer_norm is not None:
            total_input_embeded = self.input_layer_norm(total_input_embeded)

        return total_input_embeded
