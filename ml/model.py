"""Small sequence encoder + company embedding + MLP -> P(relevant)."""
import torch
import torch.nn as nn

from .config import FLIGHT_DIM, WINDOW_SIZE, COMPANY_TO_ID, EMBED_DIM

NUM_COMPANIES = len(COMPANY_TO_ID)
COMPANY_EMBED_DIM = 16
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.4  # reduce overconfident positives (fewer FP)
NEG_BIAS = -1.0  # softer: less pushed to 0 for uncertain cases (more spread in confidence)


class RelevanceModel(nn.Module):
    """
    Input: (batch, WINDOW_SIZE, FLIGHT_DIM), company_id (batch,).
    Output: (batch,) logits; sigmoid for P(relevant).
    """

    def __init__(
        self,
        flight_dim: int = FLIGHT_DIM,
        window_size: int = WINDOW_SIZE,
        num_companies: int = NUM_COMPANIES,
        company_embed_dim: int = COMPANY_EMBED_DIM,
        hidden: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.flight_dim = flight_dim
        self.window_size = window_size
        self.company_embed = nn.Embedding(num_companies, company_embed_dim)
        self.encoder = nn.GRU(
            flight_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + company_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_weights()
        # Bias toward negative: require stronger evidence to predict positive (fewer FP)
        with torch.no_grad():
            self.head[-1].bias.fill_(NEG_BIAS)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, company_id: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D), company_id: (B,) long.
        Returns (B,) logits.
        """
        _, h = self.encoder(x)
        h = h[-1]
        h = self.drop(h)
        c = self.company_embed(company_id)
        out = torch.cat([h, c], dim=1)
        return self.head(out).squeeze(-1)
