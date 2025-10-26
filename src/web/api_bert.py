from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
CORS(app)

# Modelo salvo
modelo_path = "src/modelo_bert"
tokenizer = BertTokenizer.from_pretrained(modelo_path)
model = BertForSequenceClassification.from_pretrained(modelo_path, output_attentions=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/prever_destacar", methods=["POST"])
def prever_destacar():
    data = request.json
    texto = data.get("mensagem", "")
    if not texto:
        return jsonify({"erro": "Mensagem vazia"}), 400

    # Tokenização
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = int(torch.argmax(logits, dim=1).item())
        classe = "SUSPEITA" if pred == 1 else "SEGURA"

        # Atenção da última camada
        attentions = outputs.attentions
        if attentions:
            attn = torch.mean(attentions[-1], dim=1).cpu()  # média entre cabeças
            # Atenção média por token
            importance = attn.mean(dim=1)[0].numpy()
            # Normalizar entre 0 e 1
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            # Ajustar cor pelo logit
            sign = 1 if pred == 1 else -1
            importance = importance * sign
        else:
            importance = [0.0] * inputs["input_ids"].shape[1]

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_list = []
        for t, s in zip(tokens, importance):
            # vermelho para SUSPEITA, verde para SEGURA
            if s >= 0:
                token_list.append({"text": t, "score": float(s), "color": "red"})
            else:
                token_list.append({"text": t, "score": float(-s), "color": "green"})

    return jsonify({"mensagem": texto, "classe": classe, "tokens": token_list})

# Para utilizar com a extensão "Live Server" do VS Code, por exemplo

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
