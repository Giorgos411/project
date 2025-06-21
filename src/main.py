from reconstruction.custom import custom_reconstruct
from reconstruction.t5_pipeline import t5_paraphrase
from reconstruction.grammar_tools import correct_with_gramformer, correct_with_languagetool

# Οι προτάσεις που επιλέγεις για το A
sentences = [
    "Hope you too, to enjoy it as my deepest wishes.",
    "I am very appreciated the full support of the professor, for our Springer proceedings publication"
]

print("🔸 Custom Reconstruction:")
for s in sentences:
    print("Original:", s)
    print("Reconstructed:", custom_reconstruct(s))

print("\n🔸 T5 Reconstruction:")
for s in sentences:
    print("Original:", s)
    print("Reconstructed:", t5_paraphrase(s))

print("\n🔸 Gramformer:")
for s in sentences:
    print("Original:", s)
    print("Reconstructed:", correct_with_gramformer(s))

print("\n🔸 LanguageTool:")
for s in sentences:
    print("Original:", s)
    print("Reconstructed:", correct_with_languagetool(s))
