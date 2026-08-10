import torch
from w2s.loss import WeakToStrongLoss

def test_weak_to_strong_loss_computation():
    criterion = WeakToStrongLoss(confidence_threshold=0.5)
    student_logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    weak_logits = torch.tensor([[3.0, 0.1], [0.5, 0.5]])
    
    loss = criterion(student_logits, weak_logits)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
    print("Test passed: Loss computation is stable and non-negative.")

if __name__ == "__main__":
    test_weak_to_strong_loss_computation()
