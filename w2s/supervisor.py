import torch
import torch.nn as nn

class WeakSupervisor(nn.Module):
    """Low-capacity model simulating an imperfect supervisor."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class StrongStudent(nn.Module):
    """High-capacity model simulating a frontier intelligence system."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
