import time
import os

from flask import Flask
from flask import jsonify, request
from flask_cors import CORS

from llama import Dialog, Llama
from llm import LlmApp

app = Flask(__name__)
CORS(app)


class TestClass:
    def __init__(self) -> None:
        print("class init started")
        time.sleep(5)
        print("class init completed")

        self.count = 0

    def get_value(self) -> int:
        self.count += 1
        return self.count

        print(f"Count is : {self.count}")


# test_app = TestClass()

os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# generator = Llama.build(
#     ckpt_dir="Meta-Llama-3-8B-Instruct/",
#     tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
#     max_seq_len=4096,
#     max_batch_size=6,
# )
llaapp = LlmApp()

global_count = 0


@app.route("/ask", methods=["GET", "POST"])
def dev():
    error = None

    dialogs = []

    json_data = request.get_json()['data']
    dialogs.append(json_data)
    
    # results = generator.chat_completion(
    #     dialogs,
    #     max_gen_len=None,
    #     temperature=0.6,
    #     top_p=0.9,
    # )

    # # print(f"\n{results[0]['generation']['role'].capitalize()}\n==================================\n")
    # # print(f"{results[0]['generation']['content']}")
    # # print("\n==================================\n")

    # result = {"message": f"Hi there.. I'am Jarvis. {global_count}"}

    # result = {
    #     "content": results[0]['generation']['content']
    # }
    
    # Dev
    result = {
        "content": llaapp.call(dialogs)
    }

    return jsonify(error=error, result=result)


# if __name__ == "__main__":
#     # karakara = 0

#     app.run(debug=True)
