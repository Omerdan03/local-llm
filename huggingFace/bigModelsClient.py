import requests

def getAnswer(input_string):
    url = f"http://localhost:8000/answer/{input_string}"
    response = requests.get(url)
    data = response.json()
    return data["message"]

while True:
    input_string = input("Enter a question to the model: ")
    result = getAnswer(input_string)
    print(result)