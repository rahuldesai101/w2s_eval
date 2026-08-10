import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakToStrongLoss(nn.Module):
    """
    Confidence-Gated Weak-to-Strong Supervision Loss.
    Encourages a strong student model to learn from a weak supervisor's soft labels
    while dynamically discounting uncertain predictions from the supervisor.
    """
    def __init__(self, confidence_threshold: float = 0.6, alpha: float = 0.7):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha

    def forward(
        self, 
        student_logits: torch.Tensor, 
        weak_logits: torch.Tensor, 
        ground_truth: torch.Tensor = None
    ) -> torch.Tensor:
        student_probs = F.softmax(student_logits, dim=-1)
        weak_probs = F.softmax(weak_logits, dim=-1)

        # 1. Measure supervisor confidence (Max predicted probability)
        weak_confidence, _ = torch.max(weak_probs, dim=-1)

        # 2. Mask out predictions where the supervisor is uncertain
        valid_mask = (weak_confidence >= self.confidence_threshold).float()

        # 3. Compute KL Divergence between Student and Weak Supervisor
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            weak_probs,
            reduction='none'
        ).sum(dim=-1)

        gated_weak_loss = (kl_loss * valid_mask).mean()

        # 4. Optional ground-truth cross-entropy regularization
        if ground_truth is not None:
            ce_loss = F.cross_entropy(student_logits, ground_truth)
            total_loss = (self.alpha * gated_weak_loss) + ((1 - self.alpha) * ce_loss)
        else:
            total_loss = gated_weak_loss

        return total_loss
