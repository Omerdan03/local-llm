from langchain import LLMChain, PromptTemplate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline


class Chat:

    def __init__(self, prompt, context, local_model):
        self.context = context
        self.history = []
        self._chat = LLMChain(prompt=prompt, llm=local_model)

    def parseHistory(self):
        chat_history = ""
        for message in self.history:
            if message[1] == "human":
                chat_history += f"Human: {message[0]}\n"
            if message[1] == "AI":
                chat_history += f"AI: {message[0]}\n"

        return chat_history

    def answerStoreHistory(self, qn):
        respond = self._chat.run({'context': self.context, "history": self.parseHistory(), "instruction": qn})
        if "#" in respond:
            respond = respond.split("#")[0]
        self.history.append(["human", qn])
        self.history.append(["AI", respond])
        print(f"AI: {respond}")

class LLM:
    MODEL = "mosaicml/mpt-7b-chat"
    CONSTEXT = "You are an helpful assistante in a school. You are helping a student with his homework."

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = LLM.MODEL
        self.load_model(model_name)

    def load_model(self, model_name):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "cuda:0":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                         torch_dtype=torch.float16, device_map="auto", load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )
        self.local_model = HuggingFacePipeline(pipeline=pipe)

    def get_chat(self, context):
        template = \
            """
            with the context above write a response that best complete the given instructions.
            use the chat history
            Context: {context}
            chat history:
            {history}
            Instruction: {instruction}
            Answer: """

        prompt = PromptTemplate(template=template, input_variables=["context", "history", "instruction"])

        return Chat(prompt=prompt, context=context, local_model=self.local_model)


if __name__ == "__main__":
    model = "mosaicml/mpt-7b-chat"
    context = "You are an helpfully assistant in a school. You are helping a student with his homework."

    llm = LLM(model)
    chat = llm.get_chat(context=context)
    while True:
        qn = input("Question: ")
        if qn == "exit":
            break
        chat.answerStoreHistory(qn=qn)