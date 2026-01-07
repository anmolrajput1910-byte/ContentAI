import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# LOAD DATA
# -----------------------------
with open("dataset.txt", "r", encoding="utf-8") as f:
    raw = f.read()

parts = raw.split("===SCRIPT===")
scripts = []

for p in parts:
    p = p.strip().lower()
    if len(p) > 5:
        scripts.append("<start> " + p + " <end>")

text = " ".join(scripts)
words = text.split()

# -----------------------------
# VOCAB
# -----------------------------
vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

def encode(ws):
    return [word_to_idx[w] for w in ws if w in word_to_idx]

# -----------------------------
# MODEL
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.rnn = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = MiniGPT()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# -----------------------------
# GENERATE SCRIPT (POLISHED)
# -----------------------------
def generate_script(topic, max_words=25, temperature=0.6):
    topic = topic.lower().strip()
    topic_words = topic.split()

    current = ["<start>"] + topic_words

    for _ in range(max_words):
        seq = encode(current[-10:])
        if len(seq) == 0:
            break

        seq = torch.tensor(seq).unsqueeze(0)

        with torch.no_grad():
            logits = model(seq)
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)

        next_idx = torch.multinomial(probs, 1).item()
        next_word = idx_to_word[next_idx]

        if next_word == "<end>":
            break

        current.append(next_word)

    # -----------------------------
    # POST-PROCESSING (QUALITY FIXES)
    # -----------------------------
    text = " ".join(current[1:])

    # 1️⃣ topic repeat remove (first occurrence only)
    if text.startswith(topic):
        text = text[len(topic):].strip()

    # 2️⃣ weak words filter
    bad_words = ["kabhi kabhi", "shayad", "thoda"]
    for w in bad_words:
        text = text.replace(w, "").strip()

    # 3️⃣ reel-style line break
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:8]) + "\n" + " ".join(words[8:])

    # 4️⃣ proper ending
    if not text.endswith("."):
        text += "."

    return text

torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")

