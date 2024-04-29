import torch
import torch.nn as nn


class NCF(nn.Module):
    """Neural collaborative filtering

    Parameters
    ----------
    num_users : int
        The number of users
    num_items : int
        The number of items
    num_factors : int
        The number of predictive factors of GMF model
    num_layers : int
        The number of layers in MLP model
    dropout : float
        Dropout rate between fully connected layers
    model_type : str
        Model type
        One of "MLP", "GMF", "NeuMF", "NeuMF-pre" is allowed
    pretrained_gmf : nn.Module, optional
        Pre-trained GMF weights
    pretrained_mlp : nn.Module, optional
        Pre-trained MLP weights
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int,
        num_layers: int,
        dropout: float,
        model_type: str,
        pretrained_gmf: nn.Module = None,
        pretrained_mlp: nn.Module = None,
    ):
        super(NCF, self).__init__()

        assert model_type in {
            "GMF",
            "MLP",
            "NeuMF",
            "NeuMF-pre",
        }, "Invalid model type."

        self.dropout = dropout
        self.model_type = model_type
        self.pretrained_gmf = pretrained_gmf
        self.pretrained_mlp = pretrained_mlp

        # GMF
        self.gmf_user_embedding = nn.Embedding(num_users, num_factors)
        self.gmf_item_embedding = nn.Embedding(num_items, num_factors)

        # MLP
        # Check Section 4.1 of original paper.
        # The size of last layer of mlp is equal to num_factors
        mlp_embedding_dim = num_factors * (2 ** (num_layers - 1))
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_dim)

        # stack mlp layers
        mlp_layers = []
        mlp_input_size = mlp_embedding_dim * 2
        for _ in range(num_layers):
            mlp_layers.extend(
                (
                    nn.Dropout(p=self.dropout),
                    nn.Linear(mlp_input_size, mlp_input_size // 2),
                    nn.ReLU(),
                )
            )

            mlp_input_size = mlp_input_size // 2
        self.mlp_layers = nn.Sequential(*mlp_layers)

        # For NeuMF, the last layer is the concatenate of GMF and MLP
        # So, the size of the last layer is double of the size of GMF/MLP
        if self.model_type in {"GMF", "MLP"}:
            predictive_factors = num_factors
        else:
            predictive_factors = num_factors * 2
        self.predict_layer = nn.Linear(predictive_factors, 1)

        # Initiate weights
        self._init_weight()

    def _init_weight(self):
        if self.model_type == "NeuMF-pre":
            # Load embedding weights from pre-trained model
            self.gmf_user_embedding.weight.data.copy_(
                self.pretrained_gmf.gmf_user_embedding.weight
            )
            self.gmf_item_embedding.weight.data.copy_(
                self.pretrained_gmf.gmf_item_embedding.weight
            )
            self.mlp_user_embedding.weight.data.copy_(
                self.pretrained_mlp.mlp_user_embedding.weight
            )
            self.mlp_item_embedding.weight.data.copy_(
                self.pretrained_mlp.mlp_item_embedding.weight
            )

            # Load weights of MLP layers from pre-trained model
            for (m1, m2) in zip(
                self.mlp_layers, self.pretrained_mlp.mlp_layers
            ):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # Load weights of predict layer from pre-trained model
            predict_weight = torch.cat(
                [
                    self.pretrained_gmf.predict_layer.weight,
                    self.pretrained_mlp.predict_layer.weight,
                ],
                dim=1,
            )
            predict_bias = (
                self.pretrained_gmf.predict_layer.bias
                + self.pretrained_mlp.predict_layer.bias
            )
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)

        else:
            nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
            nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)

            for layer in self.mlp_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity="sigmoid"
            )
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, user, item):
        if self.model_type == "GMF":
            concat_layer = self._gmf_forward(user, item)
        elif self.model_type == "MLP":
            concat_layer = self._mlp_forward(user, item)
        else:
            concat_layer = self._neumf_forward(user, item)

        prediction = self.predict_layer(concat_layer)
        return prediction.view(-1)

    def _mlp_forward(self, user, item) -> torch.Tensor:
        mlp_user_embedding = self.mlp_user_embedding(user)
        mlp_item_embedding = self.mlp_item_embedding(item)
        interaction = torch.cat((mlp_user_embedding, mlp_item_embedding), -1)

        return self.mlp_layers(interaction)

    def _gmf_forward(self, user, item) -> torch.Tensor:
        gmf_user_embedding = self.gmf_user_embedding(user)
        gmf_item_embedding = self.gmf_item_embedding(item)

        return gmf_user_embedding * gmf_item_embedding

    def _neumf_forward(self, user, item) -> torch.Tensor:
        return torch.cat(
            (self._gmf_forward(user, item), self._mlp_forward(user, item)),
            -1,
        )
