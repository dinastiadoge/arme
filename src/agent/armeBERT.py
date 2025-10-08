import re
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report

# -----------------------------
# 1️⃣ Carregar dataset
# -----------------------------
df = pd.read_csv("src/base/dataset.txt", sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # 0=segura, 1=suspeita

# -----------------------------
# 2️⃣ Limpeza leve dos textos
# -----------------------------
def limpar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r"http\S+|www\S+", "", texto)  # remover URLs
    texto = re.sub(r"[^a-z0-9\s]", "", texto)     # remover caracteres especiais
    texto = re.sub(r"\s+", " ", texto).strip()    # remover espaços extras
    return texto

df["message"] = df["message"].apply(limpar_texto)

# -----------------------------
# 3️⃣ Tokenização com BERT
# -----------------------------
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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# 4️⃣ Modelo, otimizador e função de perda
# -----------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# -----------------------------
# 5️⃣ Treino por 1 época (exemplo)
# -----------------------------
model.train()
for epoch in range(3):
    print(f"=== Época {epoch+1} ===")
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        print(f"Último batch Loss: {loss.item():.4f}")

        break
# -----------------------------
# 6️⃣ Avaliação
# -----------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in dataloader:
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(preds.tolist())

        break

print("\n=== Relatório de Classificação ===")
print(classification_report(y_true, y_pred, target_names=["SEGURA", "SUSPEITA"]))
