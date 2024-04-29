import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_hidden_units: int,
    ) -> None:
        super(AutoRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden_units = num_hidden_units

        self.encoder = nn.Sequential(
            nn.Linear(num_items, num_hidden_units), nn.Sigmoid()
        )

        self.decoder = nn.Sequential(nn.Linear(num_hidden_units, num_items))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder(input_data)
        return self.decoder(encoder)
