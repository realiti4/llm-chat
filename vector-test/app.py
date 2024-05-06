import random
import numpy as np

from pymilvus import DataType, MilvusClient
from pymilvus import CollectionSchema

from dev_dataset import get_dataset, get_text_embedding

# Params
COLLECTION_NAME = "karakara"
VECTOR_NAME = "my_vector"

# 1. Set up a Milvus client
client = MilvusClient(uri="http://localhost:19530")

def delete_collection(collection_name: str) -> None:
    client.drop_collection(collection_name=collection_name)

def create_schema(dim: int) -> CollectionSchema:
    """Creates an example schema for mivus"""

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="comment", datatype=DataType.VARCHAR, max_length=48000)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=1024)

    return schema


def create_collection(name: str, schema: CollectionSchema) -> None:
    # 3.3. Prepare index parameters
    index_params = client.prepare_index_params()

    # 3.4. Add indexes
    index_params.add_index(field_name="id")

    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    
    print(f"Collection {name} has created..")

delete_collection(collection_name=COLLECTION_NAME)

if not client.has_collection(collection_name=COLLECTION_NAME):
    schema = create_schema(dim=768)
    create_collection(name=COLLECTION_NAME, schema=schema)


# Github Data
data = get_dataset()

# data=[
#     {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]},
#     {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]},
#     {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592]},
#     {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]},
#     {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106]},
#     {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955]},
#     {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987]},
#     {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052]},
#     {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336]},
#     {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608]}
# ]

# 4.2. Insert data
res = client.insert(
    collection_name=COLLECTION_NAME,
    data=data
)


# Dev - score test
test_vectors = get_text_embedding("Heeeey there.")

res = client.insert(
    collection_name=COLLECTION_NAME,
    data=[{
        "id": 9999, "vector": test_vectors[0], "comment": "score test", "url": "score test"
    }]
)

print(res)

# # More data
# data = [ {
#     "id": i, 
#     "vector": [ random.uniform(-1, 1) for _ in range(5) ], 
# } for i in range(1000) ]

# # 5.2. Insert data
# res = client.insert(
#     collection_name=COLLECTION_NAME,
#     data=data[10:]
# )

# print(res)

# 6.2. Start search
query_vectors = get_text_embedding("How can I load a dataset offline?")

res = client.search(
    collection_name=COLLECTION_NAME,     # target collection
    data=query_vectors,                # query vectors
    limit=5,                           # number of returned entities
)

print(f"Result: {res}")

print("Done")