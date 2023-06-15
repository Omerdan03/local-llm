from pygpt4all.models.gpt4all import GPT4All

def new_text_callback(text):
    print(text, end="")

model = GPT4All('./models/gpt4all-converted.bin')
model.generate("What is pyramid analytics?", n_predict=600, new_text_callback=new_text_callback)