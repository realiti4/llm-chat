import os
import pickle
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from datasets import Dataset, load_dataset

device = "cuda"

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
comments_dataset = comments_dataset.add_column('id', list(range(len(comments_dataset))))

# Creating embeddings
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# model_ckpt = "BAAI/bge-m3"
model_ckpt = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# Save or load embeddings
filename = "embeddings_org.pkl"
create_embeddings = False

if create_embeddings:
    # Apply embedding to each row
    embeddings_dataset = comments_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
    )

    with open(filename, "wb") as file:  # Overwrites any existing file.
        pickle.dump(embeddings_dataset, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(filename, "rb") as file:
        embeddings_dataset = pickle.load(file)

# Using FAISS for efficient similarity search
embeddings_dataset.add_faiss_index(column="embeddings")

question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).detach().cpu().numpy()

scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=5)

# Collect samples
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"ID: {row.id}")
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()

print("Done")
