import time
import numpy as np

from typing import Optional

from pymilvus import DataType, MilvusClient
from pymilvus import CollectionSchema



class VectorDatabase:
    def __init__(self, collection_name: str, vector_dim: int, schema: Optional[CollectionSchema] = None, init_collection: bool = True) -> None:
        self.client = MilvusClient(uri="http://localhost:19530")

        self.collection_name = collection_name

        if init_collection and self.client.has_collection(collection_name=collection_name):
            self._delete_collection()

        if schema is None:
            schema = self.create_schema(dim=vector_dim)

        if not self.client.has_collection(collection_name=collection_name):
            self._create_collection(collection_name=collection_name, schema=schema)

        self.last_id = self.get_count()

    def insert(self, data: list[dict]) -> dict:
        response = self.client.insert(collection_name=self.collection_name, data=data)

        return response

    def search(self, query: list[list], limit: int = 5) -> list[list[dict]]:
        response = self.client.search(
            collection_name=self.collection_name,
            data=query,
            limit=limit,
        )

        return response

    def get_entries(self, ids: list[int]) -> list[dict]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=["role", "content"]
        )

        return res
    
    def get_count(self) -> int:
        res = self.client.query(
            collection_name=self.collection_name,
            output_fields=["count(*)"]
        )

        return res[0]['count(*)']

    def create_schema(self, dim: int) -> CollectionSchema:
        """Creates an example schema for mivus"""

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)

        return schema

    def _create_collection(self, collection_name: str, schema: CollectionSchema) -> None:
        index_params = self.client.prepare_index_params()

        index_params.add_index(field_name="id")

        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")

        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

        print(f"Collection {collection_name} has created..")

    def _delete_collection(self) -> None:
        assert self.client.has_collection(collection_name=self.collection_name), "Collection can not be found!"

        self.client.drop_collection(collection_name=self.collection_name)
        print(f"Collection {self.collection_name} is deleted..")


if __name__ == "__main__":
    db = VectorDatabase(collection_name="karakara", vector_dim=768)

    print("Done")