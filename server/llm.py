import os

from llama import Llama, Dialog
from llama.tokenizer import Tokenizer


class LlmApp:
    def __init__(self) -> None:
        self.max_seq_len = 4096

        self.generator = Llama.build(
            ckpt_dir="Meta-Llama-3-8B-Instruct/",
            tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
            max_seq_len=self.max_seq_len,
            max_batch_size=6,
        )

    def calc_token_size(self, dialogs) -> int:
        """
            Dev - calculate approximated token count
        """

        word_count = 0
        ratio = 1.4

        for dialog in dialogs[0]:
            word_count += len(dialog['content'].split(' '))

        return word_count * ratio

    def call(self, dialogs: list[Dialog]) -> str:
        token_count = self.calc_token_size(dialogs)

        # if token_count > self.max_seq_len:

        #     print('token size has surpassed')

        #     while token_count > self.max_seq_len:
        #         dialogs[0].pop(0)
        #         token_count = self.calc_token_size(dialogs)

        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=0.6,
            top_p=0.9,
        )

        return results[0]["generation"]["content"]


if __name__ == "__main__":
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dialogs = [[{"role": "user", "content": "Hi there"}]]

    dev_app = LlmApp()

    response = dev_app.call(dialogs)

    print("Done")
