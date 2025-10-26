import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Carregar modelo e tokenizer treinados
modelo_path = "src/modelo_bert"
tokenizer = BertTokenizer.from_pretrained(modelo_path)
model = BertForSequenceClassification.from_pretrained(modelo_path)
model.eval()

# 2. Função de previsão
def prever_mensagem(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "SUSPEITA" if pred == 1 else "SEGURA"

# 3. Teste
#print(prever_mensagem("Hey, are we still meeting at 3 pm?"))
#print(prever_mensagem("Hi, what's up? I'm waiting you for the meeting."))

mensagens_ham = [
    "Hey, are we still meeting at 3 pm?",
    "Don't forget to bring the documents tomorrow.",
    "Can you pick up some milk on your way home?",
    "I'll call you after the meeting.",
    "Happy birthday! Hope you have a great day.",
    "Just finished the report, sending it now.",
    "Are you free for lunch today?",
    "Thanks for your help yesterday.",
    "Remember to submit your timesheet.",
    "Can you send me the presentation slides?",
    "I'll meet you at the park later.",
    "Let's catch up over coffee this week.",
    "Your package has been delivered.",
    "Don't forget the team meeting tomorrow.",
    "I'm on my way, see you soon.",
    "Did you get my email?",
    "Thanks for the invitation, I'll be there.",
    "Please confirm your attendance.",
    "The meeting has been rescheduled to 4 pm.",
    "Could you review this document and send feedback?"
]

mensagens_spam = [
    "Congratulations! You won a free iPhone, click here!",
    "Claim your $1000 gift card now!",
    "You’ve been selected for a cash prize, act fast!",
    "Urgent! Verify your account to avoid suspension.",
    "This is not a scam, click to get your reward!",
    "Get cheap meds online without prescription.",
    "Work from home and earn $5000/week!",
    "You won a lottery! Claim your money here.",
    "Free trial available, sign up now!",
    "Limited time offer! Don’t miss out!",
    "Act now to unlock exclusive benefits.",
    "Your account has been compromised, click to fix.",
    "Earn money fast with this simple trick.",
    "Get a new credit card instantly!",
    "Special offer: Buy one, get one free!",
    "You are eligible for a loan approval today.",
    "Click here to receive your free gift.",
    "Hot singles waiting to meet you online!",
    "Update your payment info to avoid cancellation.",
    "Claim your free vacation now!"
]

print("===== MENSAGENS SEGURAS =====")
for msg in mensagens_ham:
    print(f"{msg} -> {prever_mensagem(msg)}")

print("\n===== MENSAGENS SPAM =====")
for msg in mensagens_spam:
    print(f"{msg} -> {prever_mensagem(msg)}")
