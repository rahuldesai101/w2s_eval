import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Load Tokenizer and Models
weak_model_id = "distilbert/distilbert-base-uncased"
strong_model_id = "bert-base-uncased"

print("==================================================")
print("Loading Models for Weak-to-Strong Evaluation")
print("==================================================")
tokenizer = AutoTokenizer.from_pretrained(weak_model_id)
weak_supervisor = AutoModelForSequenceClassification.from_pretrained(weak_model_id, num_labels=2)
strong_student = AutoModelForSequenceClassification.from_pretrained(strong_model_id, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weak_supervisor.to(device)
strong_student.to(device)

# 2. Load IMDb Datasets (Train and Test Split)
print("\nLoading IMDb dataset...")
train_dataset = load_dataset("imdb", split="train[:500]")
test_dataset = load_dataset("imdb", split="test[:200]")

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_encoded = train_dataset.map(tokenize_fn, batched=True)
test_encoded = test_dataset.map(tokenize_fn, batched=True)

train_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_encoded, batch_size=8, shuffle=True)
test_loader = DataLoader(test_encoded, batch_size=8, shuffle=False)

# 3. Accuracy Evaluation Function
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 4. Confidence-Gated Alignment Loss
def confidence_gated_loss(student_logits, weak_logits, confidence_threshold=0.6):
    weak_probs = F.softmax(weak_logits, dim=-1)
    weak_confidence, _ = torch.max(weak_probs, dim=-1)
    valid_mask = (weak_confidence >= confidence_threshold).float()

    kl_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        weak_probs,
        reduction='none'
    ).sum(dim=-1)

    return (kl_loss * valid_mask).mean()

# 5. Evaluate Baseline Weak Supervisor
print("\nEvaluating Baseline Weak Supervisor Accuracy...")
acc_weak = evaluate_accuracy(weak_supervisor, test_loader)

# 6. Fine-tune Strong Student on Weak Supervision
optimizer = torch.optim.AdamW(strong_student.parameters(), lr=2e-5)
weak_supervisor.eval()
strong_student.train()

print("\nStarting Weak-to-Strong Fine-Tuning...")
for epoch in range(1, 3):
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            weak_outputs = weak_supervisor(input_ids, attention_mask=attention_mask)
            weak_logits = weak_outputs.logits

        student_outputs = strong_student(input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        loss = confidence_gated_loss(student_logits, weak_logits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:02d}/02 | Alignment Loss: {avg_loss:.6f}")

# 7. Evaluate Aligned Strong Student
print("\nEvaluating Aligned Strong Student Accuracy...")
acc_w2s = evaluate_accuracy(strong_student, test_loader)

# 8. Compute Metrics & Display Results
print("\n==================================================")
print("FINAL BENCHMARK RESULTS")
print("==================================================")
print(f"Weak Supervisor Accuracy (DistilBERT): {acc_weak * 100:.2f}%")
print(f"Aligned Strong Student Accuracy (BERT):   {acc_w2s * 100:.2f}%")
print(f"Weak-to-Strong Net Gain:                +{(acc_w2s - acc_weak) * 100:.2f}%")
print("==================================================")
