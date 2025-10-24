# Interpretabilidade.py
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# Carregar modelo e tokenizer já salvos
modelo_path = "src/modelo_bert"
tokenizer = BertTokenizer.from_pretrained(modelo_path)
model = BertForSequenceClassification.from_pretrained(modelo_path, output_attentions=True)
model.eval()

# Função de destaque
def destacar_tokens(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    attentions = outputs.attentions  # lista de tensores (camadas x cabeças x tokens x tokens)

    # Média das atenções das últimas camadas (foco principal)
    attn = torch.mean(attentions[-1], dim=(0, 1)).detach().numpy()  # média entre cabeças e camadas
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Normalizar pesos
    importance = attn.mean(axis=0)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    # Visualização textual simples
    print("\n📖 Texto analisado:\n")
    for token, score in zip(tokens, importance):
        intensity = int(score * 255)
        print(f"\033[48;2;{255-intensity};{255-intensity};255m {token} \033[0m", end=" ")
    print("\n")

# Exemplo de uso
texto = "Congratulations! You've won a free ticket. Call now!"
destacar_tokens(texto)
