from transformers import pipeline

from transformers import pipeline, T5Tokenizer

model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)

paraphraser = pipeline(
    "text2text-generation",
    model=model_name,
    tokenizer=tokenizer
)

def t5_paraphrase(text: str) -> str:
    output = paraphraser("paraphrase: " + text, max_length=128, do_sample=True)
    return output[0]['generated_text']
