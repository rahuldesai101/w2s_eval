import torch
from torch.utils.data import DataLoader, TensorDataset
from w2s.supervisor import WeakSupervisor, StrongStudent
from w2s.trainer import W2STrainer

def main():
    print("==================================================")
    print("Initializing Weak-to-Strong Alignment Benchmark")
    print("==================================================")

    input_dim = 128
    num_samples = 1000
    batch_size = 64

    X = torch.randn(num_samples, input_dim)
    Y = torch.randint(0, 2, (num_samples,))

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    supervisor = WeakSupervisor(input_dim=input_dim, hidden_dim=16)
    student = StrongStudent(input_dim=input_dim, hidden_dim=512)

    trainer = W2STrainer(student=student, supervisor=supervisor, lr=1e-3, confidence_threshold=0.6)

    print(f"Training on device: {trainer.device}")
    for epoch in range(1, 6):
        avg_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch:02d}/05 | Gated Alignment Loss: {avg_loss:.6f}")

    print("==================================================")
    print("Experiment completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
