from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('InstaDeepAI/nucleotide-transformer-500m-human-ref')
print('Success!')
