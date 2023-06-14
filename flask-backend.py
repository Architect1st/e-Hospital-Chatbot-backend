from flask import Flask, jsonify, request
from flask_cors import CORS
from llama_cpp import Llama
import re

llm = Llama(model_path="./vicuna-7b-1.1.ggmlv3.q4_K_S.bin")
# llm = Llama(model_path="./ggjtv1-model-q4_0.bin")
# llm = Llama(model_path="./Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_K_S.bin")

app = Flask(__name__)
CORS(app)

def extract_last_answer(dialogs):
    answers = re.findall(r"A:\s*([^Q]+)", dialogs)
    if answers:
        last_answer = answers[-1].strip()
        return last_answer
    else:
        return None


@app.route('/', methods=['POST'])
def chatbot():
    if request.method == "POST":
        user_input = request.json.get('userObject').get('userInput')
        # print(f"User Input: {user_input}")
        
        prompt = "Q: You are now a medical doctor, and I am a patient. I will describe my conditions and symptoms and you will give me medical suggestions. You may ask me to do further medical testings if you are not sure about my symptoms. Ok? A: Sure. Q: "
        human_input = prompt + user_input["message"] + " A:"
        # print(human_input)
        output = llm(human_input, max_tokens=128, stop=["Q:"], echo=True) #"\n"
        print(output)
        response = output["choices"][0]["text"]
        
        return jsonify({"response": extract_last_answer(response)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
