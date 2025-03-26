from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to("mps")  
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def rephrase_text(input_text):
    input_ids = t5_tokenizer.encode("paraphrase: " + input_text, return_tensors="pt").to("mps")
    output_ids = t5_model.generate(input_ids, max_length=100, num_return_sequences=1)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

dataset_text = "The quick brown fox jumps over the lazy dog." 
ai_generated_text = rephrase_text(dataset_text)

print("AI-Generated Text:", ai_generated_text)

