from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j")

prompt = "Somatic hypermutation allows the immune system to"
inputs = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0]))
