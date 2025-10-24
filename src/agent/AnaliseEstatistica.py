# AnaliseEstatistica.py
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_scheduler, BertTokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from armeBERT import dataset, SMSDataset  # importa o dataset e classe já criada

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros a testar
batch_sizes = [8, 16]
learning_rates = [2e-5, 3e-5]
epochs = [2, 3]
num_folds = 3  # k-fold cross validation

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Função de treino + avaliação
def train_and_eval(model, train_loader, val_loader, lr, num_epochs):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Época {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")

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
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1


# Validação cruzada
kfold = KFold(n_splits=num_folds, shuffle=True)
best_score = 0
best_params = None

for batch_size in batch_sizes:
    for lr in learning_rates:
        for num_epochs in epochs:
            fold_accuracies, fold_f1s = [], []
            print(f"\n🔍 Testando combinação: batch={batch_size}, lr={lr}, epochs={num_epochs}")

            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                print(f"\n--- Fold {fold+1}/{num_folds} ---")
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

                model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
                acc, f1 = train_and_eval(model, train_loader, val_loader, lr, num_epochs)
                fold_accuracies.append(acc)
                fold_f1s.append(f1)

            mean_acc = np.mean(fold_accuracies)
            mean_f1 = np.mean(fold_f1s)
            print(f"Resultado médio → Acurácia={mean_acc:.4f}, F1={mean_f1:.4f}")

            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = (batch_size, lr, num_epochs)

# Melhor combinação
print("\n==============================")
print(f"Melhor combinação encontrada:")
print(f"Batch size: {best_params[0]}")
print(f"Learning rate: {best_params[1]}")
print(f"Epochs: {best_params[2]}")
print(f"F1 médio: {best_score:.4f}")
print("==============================")

# Salvar os melhores parâmetros
torch.save(best_params, "src/modelo_bert/melhores_parametros.pt")
print("\n Parâmetros salvos!")
