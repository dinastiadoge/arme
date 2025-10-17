import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# ===============================
# 1️⃣ Carregar e preparar o dataset
# ===============================
# Ler o arquivo bruto linha por linha
with open("src/base/dataset.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()

linhas_limpo = []
for l in linhas:
    l = l.strip()
    # Substitui qualquer sequência de espaços ou tabs por um único tab
    l = re.sub(r"[\t ]+", "\t", l, count=1)  # count=1 para substituir só o primeiro separador
    linhas_limpo.append(l)

# Salvar de volta
with open("src/base/dataset_limpo.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_limpo))


df = pd.read_csv("src/base/dataset_limpo.txt", sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # 0 = segura, 1 = suspeita

# ===============================
# 2️⃣ Limpeza leve dos textos
# ===============================
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-z0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Converter NaN para string vazia antes da limpeza
df["message"] = df["message"].fillna("").astype(str) # Garantir que todas as mensagens sejam strings válidas
df["message"] = df["message"].apply(limpar_texto)

# ===============================
# 3️⃣ Tokenização com BERT
# ===============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = SMSDataset(df["message"].tolist(), df["label"].tolist(), tokenizer)

dataset.labels = dataset.labels.long()

#with open("src/base/dataset.txt", "r", encoding="utf-8") as f:
 #   first_line = f.readline()
  #  print(repr(first_line))


# Dividir dataset em treino (80%) e teste (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ===============================
# 4️⃣ Criar e configurar modelo BERT
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 2
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ===============================
# 5️⃣ Treinamento e avaliação
# ===============================
for epoch in range(num_epochs):
    print(f"\n===== Época {epoch + 1}/{num_epochs} =====")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Treinando"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Loss médio da época: {avg_loss:.4f}")

    # Avaliação após cada época
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nRelatório detalhado:")
    print(classification_report(y_true, y_pred, target_names=["SEGURA", "SUSPEITA"], digits=4))