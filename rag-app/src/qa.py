# qa.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QA:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    def answer(self, question, context):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
