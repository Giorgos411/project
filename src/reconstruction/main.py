import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from reconstruction.custom import custom_reconstruct
from reconstruction.t5_pipeline import t5_paraphrase
from reconstruction.grammar_tools import correct_with_gramformer, correct_with_languagetool

# 🔹 NLP Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Raw Input
sentences = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to  the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from  the professor, to show me, this, a couple of days ago.  I am very appreciated  the full support of the professor, for our Springer proceedings publication",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"
]

# 🔹 Reconstruct with all methods
results = {
    "Original": sentences,
    "Custom": [custom_reconstruct(s) for s in sentences],
    "T5": [t5_paraphrase(s) for s in sentences],
    "Gramformer": [correct_with_gramformer(s) for s in sentences],
    "LanguageTool": []
}

# LanguageTool με delay για αποφυγή limits
for s in sentences:
    try:
        results["LanguageTool"].append(correct_with_languagetool(s))
    except Exception as e:
        results["LanguageTool"].append(f"ERROR: {e}")
    time.sleep(5)

# 🔹 Εκτύπωση Αποτελεσμάτων Παραδοτέου 1
print("\n🔸 Ανακατασκευή Όλων των Κειμένων (A,B):")
for i, original in enumerate(sentences):
    print(f"\n--- Κείμενο {i+1} ---")
    print("Original:", original)
    for method, reconstructions in results.items():
        if method != "Original":
            print(f"{method} Reconstruction:\n{reconstructions[i]}\n")

# 🔹 Cosine Similarity & Embeddings Analysis
print("\n🔸 Cosine Similarity μεταξύ Original και Reconstructed:")

for i, original in enumerate(sentences):
    print(f"\n--- Κείμενο {i+1} ---")
    emb_orig = model.encode([original])[0]
    for method in ["Custom", "T5", "Gramformer", "LanguageTool"]:
        recon = results[method][i]
        emb_recon = model.encode([recon])[0]
        sim = cosine_similarity([emb_orig], [emb_recon])[0][0]
        print(f"{method}: {sim:.4f}")

# 🔹 PCA Οπτικοποίηση (1ο Κείμενο)
print("\n🔸 Οπτικοποίηση PCA των Embeddings (1ο Κείμενο):")

embeddings = []
labels = []

for method in results:
    embeddings.append(model.encode([results[method][0]])[0])
    labels.append(method)

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(labels))

for i in range(len(reduced)):
    plt.scatter(reduced[i][0], reduced[i][1], label=labels[i], s=100, color=palette[i])
    plt.text(reduced[i][0]+0.02, reduced[i][1]+0.02, labels[i], fontsize=12)

plt.title("PCA των Embeddings (1ο Κείμενο)", fontsize=14)
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.show()
