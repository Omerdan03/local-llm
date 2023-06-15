from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI

app = FastAPI()

MODEL = "NousResearch/Nous-Hermes-13b"
MODEL = "gpt2"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir="D:\.cache\huggingface")
model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir="D:\.cache\huggingface")
print("Model loaded.")

@app.get("/answer/{prompt}")
def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    respond = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"message": respond}

# Run this file with uvicorn:n bigMo
# uvicorn bigModelsServer:app --reload