"""
structeff/siamese.py — contrastive (Siamese) encoder used by StructEff.

This class is required to unpickle `model_cl` from the saved model bundle.
The architecture and forward() are taken verbatim from the training script
(hybrid_model ... VERSION2), so the restored weights and the contrastive
transform behave exactly as they did at training time.

predict.py calls the model as `model_cl(X_tensor)`, which returns the
256-dim transformed embedding that is concatenated with the input features
before XGBoost.
"""
import torch.nn as nn


class Siamese(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
        )

    def forward(self, x):
        return self.net(x)
