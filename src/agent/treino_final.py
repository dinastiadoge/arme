import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, get_scheduler
from tqdm import tqdm
from armeBERT import dataset  # seu dataset já processado, SMSDataset incluso

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_path = "src/modelo_bert"

# Hiperparâmetros a testar
batch_sizes = [8, 16]
learning_rates = [2e-5, 3e-5]
epochs_list = [2, 3]
num_folds = 3

# Função de treino + avaliação
def train_and_eval(model, train_loader, val_loader, lr, num_epochs):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_loader)
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nÉpoca {epoch+1}/{num_epochs}...")
        for batch in tqdm(train_loader, desc="Treinando", ncols=80):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Loss médio: {total_loss/len(train_loader):.4f}")

    # Avaliação
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred)
    return acc, f1

# ===== Validação cruzada para escolher melhores hiperparâmetros =====
best_score = 0
best_params = None

print("\n=== Iniciando validação cruzada ===")
for batch_size in batch_sizes:
    for lr in learning_rates:
        for num_epochs in epochs_list:
            print(f"\nTestando combinação: batch={batch_size}, lr={lr}, epochs={num_epochs}")
            fold_accuracies, fold_f1s = [], []
            kfold = KFold(n_splits=num_folds, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                print(f"\n--- Fold {fold+1}/{num_folds} ---")
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=2
                ).to(device)
                acc, f1 = train_and_eval(model, train_loader, val_loader, lr, num_epochs)
                fold_accuracies.append(acc)
                fold_f1s.append(f1)
                print(f"Fold {fold+1} → Acc: {acc:.4f}, F1: {f1:.4f}")
            mean_f1 = np.mean(fold_f1s)
            print(f"Média F1 dessa combinação: {mean_f1:.4f}")
            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = (batch_size, lr, num_epochs)

print("\n=== Melhor combinação encontrada ===")
print(f"Batch size: {best_params[0]}")
print(f"Learning rate: {best_params[1]}")
print(f"Epochs: {best_params[2]}")
print(f"F1 médio: {best_score:.4f}")

# Salvar melhores parâmetros
torch.save(best_params, os.path.join(modelo_path, "melhores_parametros.pt"))
print("\nParâmetros salvos!")

# ===== Treinamento final com todo o dataset =====
print("\n=== Treinando modelo final no dataset completo ===")
final_loader = DataLoader(dataset, batch_size=best_params[0], shuffle=True)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(device)
train_and_eval(model, final_loader, final_loader, best_params[1], best_params[2])

# Salvar modelo e tokenizer
os.makedirs(modelo_path, exist_ok=True)
model.save_pretrained(modelo_path)
BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained(modelo_path)
print(f"\nModelo e tokenizer salvos em: {modelo_path}")
