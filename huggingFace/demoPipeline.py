from transformers import pipeline


def sentimentAnalysis():
    model = "distilbert-base-uncased-finetuned-sst-2-english"

    nlp_model = pipeline(task="sentiment-analysis", model=model)
    data = ["I love you", "I hate you"]

    print(nlp_model(data))

def textGeneration():
    model = "gpt2"
    # model = "facebook/opt-125m"

    generator_model = pipeline(task='text-generation', model=model)
    print(generator_model("My name is Teven and I am a student at the University of California, Berkeley. I am", max_length=50, do_sample=True))

def questionAnswering():
    # model = "deepset/roberta-base-squad2"
    model = "facebook/opt-125m"

    QA_input = {
        'question': 'what is a zebra?',
        'context': ' '
    }

    # QA_input = {
    #     'question': 'How many brothers does ron have?',
    #     'context': 'Tommy has 3 sons, Ron, John and Harry.'
    # }

    answering_model = pipeline(task='question-answering', model=model)
    print(answering_model(QA_input))



if __name__ == "__main__":
    # sentimentAnalysis()
    textGeneration()
    # questionAnswering()

