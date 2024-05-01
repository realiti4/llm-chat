import torch

from typing import List, Optional

# from llama import Dialog, Llama
from transformers import AutoTokenizer, AutoModel


class Embeddings:
    def __init__(
        self, model_ckpt: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", device: str = "cuda"
    ) -> None:
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(device)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list) -> torch.Tensor:
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        
        return self.cls_pooling(model_output)
    
    def format_dialogs(self, dialogs: list[dict[str, str]]) -> list[str]:
        output = []

        for item in dialogs:
            formatted = f"{item["role"]}:\n{item["content"]}"

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

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    embeddings = Embeddings()

    dialogs: List[Dialog] = [[]]

    while True:
        user_content = input("Ask: ")

        user_data = {"role": "user", "content": user_content}

        test = embeddings.format_dialogs(user_data)

        dialogs[0].append(user_data)

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        print("de")

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            print("\n==================================\n")

def dev():
    embeddings = Embeddings()

    example_dialogs = [{
        "role": "user",
        "content": "This is an example question.."
    }]

    results = embeddings.format_dialogs(example_dialogs)



if __name__ == "__main__":
    # main(
    #     ckpt_dir="",
    #     tokenizer_path="",
    #     temperature=0.6,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_batch_size=4,
    #     max_gen_len=None,
    # )

    dev()

    # fire.Fire(main)
