import os
import torch
from armeBERT import df, SMSDataset  # Dataset já processado
from AnaliseEstatistica import batch_sizes, learning_rates, epochs, KFold, np, train_and_eval
from Interpretabilidade import destacar_tokens
from torch.utils.data import DataLoader, Subset
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_path = "src/modelo_bert"

# Treinamento e validação cruzada
print(" Iniciando validação cruzada para melhores hiperparâmetros...")
from AnaliseEstatistica import dataset  # importa o dataset do armeBERT.py

num_folds = 3
kfold = KFold(n_splits=num_folds, shuffle=True)
best_score = 0
best_params = None

for batch_size in batch_sizes:
    for lr in learning_rates:
        for num_epochs in epochs:
            fold_accuracies, fold_f1s = [], []
            print(f"\nTestando: batch={batch_size}, lr={lr}, epochs={num_epochs}")

            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
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
            print(f"Média → Acurácia={mean_acc:.4f}, F1={mean_f1:.4f}")

            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = (batch_size, lr, num_epochs)

# Salvar melhores parâmetros
os.makedirs(modelo_path, exist_ok=True)
torch.save(best_params, os.path.join(modelo_path, "melhores_parametros.pt"))
print(f"\n Melhores parâmetros: batch={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]} | F1={best_score:.4f}")

# Treinar modelo final com melhores parâmetros
print("\n Treinando modelo final com melhores hiperparâmetros...")
batch_size, lr, num_epochs = best_params

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

from AnaliseEstatistica import train_and_eval
# Treinar sem validação, apenas para salvar o modelo final
train_and_eval(model, train_loader, train_loader, lr, num_epochs)  # aqui o val_loader é igual ao train_loader apenas para rodar o treino

# Salvar modelo e tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.save_pretrained(modelo_path)
tokenizer.save_pretrained(modelo_path)
print(f" Modelo e tokenizer salvos em {modelo_path}")

# Interpretabilidade
print("\n Analisando tokens importantes de exemplo...")
texto_exemplo = "Congratulations! You've won a free ticket. Call now!"
destacar_tokens(texto_exemplo)
