import os
import pickle
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from datasets import Dataset, load_dataset

device = "cuda"

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# model_ckpt = "BAAI/bge-m3"
model_ckpt = "BAAI/bge-base-en-v1.5"


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(model, tokenizer, text_list) -> torch.Tensor:
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def get_dataset():
    filename = "embeddings.pkl"
    create_embeddings = False

    if not create_embeddings:
        with open(filename, "rb") as file:
            output = pickle.load(file)

        return output

    issues_dataset = load_dataset("lewtun/github-issues", split="train")

    issues_dataset = issues_dataset.filter(lambda x: (x["is_pull_request"] is False and len(x["comments"]) > 0))

    # Remove unnecessary columns
    columns = issues_dataset.column_names
    columns_to_keep = ["title", "body", "html_url", "comments"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    issues_dataset = issues_dataset.remove_columns(columns_to_remove)

    # Match each comment with title and body etc.
    issues_dataset.set_format("pandas")
    df = issues_dataset[:]

    comments_df = df.explode("comments", ignore_index=True)  # expandes each comment with remaining columns

    comments_dataset = Dataset.from_pandas(comments_df)

    comments_dataset = comments_dataset.map(lambda x: {"comment_length": len(x["comments"].split())})
    comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)  # filter short comments

    # concatenate the issue title, description, and comments together
    def concatenate_text(examples):
        return {"text": examples["title"] + " \n " + examples["body"] + " \n " + examples["comments"]}

    comments_dataset = comments_dataset.map(concatenate_text)

    # Creating embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    comments = []
    urls = []
    embeddings = []

    for row in comments_dataset:
        comments.append(row["comments"])
        urls.append(row["html_url"])
        embeddings.append(get_embeddings(model=model, tokenizer=tokenizer, text_list=row["text"]).detach().cpu().numpy()[0])

    # embeddings = len([i.tolist() for i in embeddings])

    output = []

    for i, (comment, url, embedding) in enumerate(zip(comments, urls, embeddings)):
        output.append({"id": i, "comment": comment[:12000], "url": url, "vector": embedding})

    with open(filename, "wb") as file:  # Overwrites any existing file.
        pickle.dump(output, file, pickle.HIGHEST_PROTOCOL)

    return output

def get_text_embedding(text: str) -> list:
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    question_embedding = get_embeddings(model=model, tokenizer=tokenizer, text_list=[text]).detach().cpu().numpy()

    return question_embedding.tolist()

if __name__ == "__main__":
    get_dataset()
    # get_text_embedding("How can I load a dataset offline?")
