"""
models.py
---------
LSTM mimarisi — 05_train.py'deki PM25LSTM ile birebir ayni.
state_dict yuklendiginde tum agirlik anahtarlari eslemeli.
"""

from __future__ import annotations

import torch.nn as nn

N_FEATURES = 18


class PM25LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=N_FEATURES, hidden_size=64,
                             num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32,
                             num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(32, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out)
        out    = out[:, -1, :]
        out    = self.relu(self.fc1(out))
        out    = self.fc2(out)
        return out.squeeze(-1)
