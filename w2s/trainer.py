import torch
import torch.optim as optim
from .loss import WeakToStrongLoss

class W2STrainer:
    def __init__(
        self, 
        student: torch.nn.Module, 
        supervisor: torch.nn.Module, 
        lr: float = 1e-3, 
        confidence_threshold: float = 0.6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.supervisor = supervisor.to(self.device)
        self.criterion = WeakToStrongLoss(confidence_threshold=confidence_threshold)
        self.optimizer = optim.AdamW(self.student.parameters(), lr=lr)

    def train_epoch(self, dataloader) -> float:
        self.student.train()
        self.supervisor.eval()
        total_loss = 0.0

        for x_batch, _ in dataloader:
            x_batch = x_batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                weak_logits = self.supervisor(x_batch)

            student_logits = self.student(x_batch)
            loss = self.criterion(student_logits, weak_logits)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
