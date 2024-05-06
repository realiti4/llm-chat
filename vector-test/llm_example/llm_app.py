import os
import torch

from typing import List, Optional

from llama import Dialog, Llama
from transformers import AutoTokenizer, AutoModel

from client import VectorDatabase

os.environ['LOCAL_RANK'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'


class Embeddings:
    def __init__(
        self, model_ckpt: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", device: str = "cuda"
    ) -> None:
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(device)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list: list[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)

        return self.cls_pooling(model_output)

    def format_dialogs(self, dialogs: list[dict[str, str]]) -> list[str]:
        output = []

        for item in dialogs:
            formatted = f"{item["role"]}:\n{item["content"]}\n"

            output.append(formatted)

        return output


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    embeddings = Embeddings(model_ckpt="BAAI/bge-base-en-v1.5")
    db = VectorDatabase(collection_name="karakara", vector_dim=768, init_collection=False)
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [[{
        "role": "system",
        "content": "Your name is Jarvis."
    }]]

    while True:
        user_content = input("Ask: ")

        # Get stored data
        user_content_vector = embeddings.get_embeddings([user_content]).detach().cpu().numpy()
        related_data = db.search(user_content_vector.tolist())

        if len(related_data[0]) > 0:
            ids = [i['id'] for i in related_data[0]]

            related_entries = db.get_entries(ids=ids)
            rag = embeddings.format_dialogs(related_entries)

            dialogs[0][0]["content"] = """Your name is Jarvis. Add a list of brief subjects at the end of your responses that you have talked about so far. You can also use the information from your previous chats below if you need them. If you don't, answer as usual. Previous chats:\n"""

            for item in rag:
                dialogs[0][0]["content"] += item
            
        user_data = {"role": "user", "content": user_content}

        dialogs[0].append(user_data)

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        print(f"\n{results[0]['generation']['role'].capitalize()}\n==================================\n")
        print(f"{results[0]['generation']['content']}")
        print("\n==================================\n")

        dialogs[0].append(results[0]['generation'])

        # Insert Vectors
        res = insert_vectors(db, embeddings, [user_data, results[0]['generation']])

        print(res)


def insert_vectors(db: VectorDatabase, embeddings: Embeddings, data: list[dict]):
    # Get embeddings
    vectors = embeddings.get_embeddings([i["content"] for i in data])
    vectors = vectors.detach().cpu().numpy()

    # Insert data
    insert_data = []

    for i in range(len(data)):
        insert_data.append(
            {
                "id": db.last_id,
                "role": data[i]["role"],
                "content": data[i]["content"],
                "vector": vectors[i],
            }
        )

        db.last_id += 1

    res = db.insert(insert_data)

    return res


def dev():
    db = VectorDatabase(collection_name="karakara", vector_dim=768)
    embeddings = Embeddings()

    example_dialogs = [
        {"role": "user", "content": "This is an example question.."},
        {"role": "assistant", "content": "Nice to meet you.."},
    ]

    res = insert_vectors(db, embeddings, example_dialogs)

    print(res)

    # results = embeddings.format_dialogs(example_dialogs)

    print("Done")


if __name__ == "__main__":
    main(
        ckpt_dir="Meta-Llama-3-8B-Instruct/",
        tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
        temperature=0.6,
        top_p=0.9,
        max_seq_len=4096,
        max_batch_size=4,
        max_gen_len=None,
    )

    # dev()

