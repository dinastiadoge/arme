import re
import unicodedata
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Garantir recursos necessários do NLTK
nltk.download('punkt')

# Função de limpeza
def limpar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Função de tokenização
def tokenizar_texto(texto):
    return texto.split()

# 1. Ler o dataset (ajustado para o formato do projeto)
df = pd.read_csv("src/base/dataset.txt", sep="\t", header=None, names=["label", "mensagem"])

# 2. Converter rótulos para formato binário
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 3. Aplicar limpeza e tokenização
df['mensagem_limpa'] = df['mensagem'].apply(limpar_texto)
df['tokens'] = df['mensagem_limpa'].apply(tokenizar_texto)

# 4. Vetorização (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['mensagem_limpa'])
y = df['label']

# 5. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Treinar o modelo de Regressão Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Fazer previsões
y_pred = model.predict(X_test)

# 8. Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=["HAM", "SPAM"]))