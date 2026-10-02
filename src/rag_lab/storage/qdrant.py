import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    SearchParams,
    VectorParams,
)

from rag_lab.config import IndexConfig

PAYLOAD_INDEXES = ("doc_id", "modality", "source_file")
_NAMESPACE = uuid.UUID("6f1d3a52-6d0b-4e0e-9a43-5b7d1c1f0a11")


def to_point_id(chunk_id: str) -> str:
    """Qdrant ids must be ints or UUIDs. uuid5 keeps them deterministic, so re-ingesting overwrites."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(self, url: str, grpc_port: int = 6334):
        self.client = QdrantClient(url=url, prefer_grpc=True, grpc_port=grpc_port)

    def ensure_collection(self, name: str, dimension: int, index: IndexConfig) -> None:
        """Create the collection if missing; refuse to reuse one with a different vector size."""
        if self.client.collection_exists(name):
            params = self.client.get_collection(name).config.params.vectors
            if params.size != dimension:
                raise ValueError(
                    f"Collection '{name}' has vector size {params.size}, expected {dimension}"
                )
            return
        self.client.create_collection(
            name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=index.hnsw_m, ef_construct=index.hnsw_ef_construct),
        )
        for field in PAYLOAD_INDEXES:
            self.client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        batch_size: int = 256,
    ) -> int:
        """`ids` are already point ids (see to_point_id). Waits for each batch so timings are honest."""
        for i in range(0, len(ids), batch_size):
            points = [
                PointStruct(id=pid, vector=vec, payload=pay)
                for pid, vec, pay in zip(
                    ids[i : i + batch_size],
                    vectors[i : i + batch_size],
                    payloads[i : i + batch_size],
                )
            ]
            self.client.upsert(collection, points, wait=True)
        return len(ids)

    def delete_document(self, collection: str, doc_id: str) -> None:
        self.client.delete(
            collection,
            FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
        hnsw_ef: int | None = None,
    ) -> list[ScoredPoint]:
        """Nearest points by cosine. With the cosine metric, `score` is the similarity itself."""
        if not self.client.collection_exists(collection):
            raise ValueError(f"Collection '{collection}' does not exist. Ingest a document first.")
        query_filter = (
            Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])
            if filters
            else None
        )
        response = self.client.query_points(
            collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None,
            with_payload=True,
        )
        return response.points

    def has_document(self, collection: str, doc_id: str) -> bool:
        if not self.client.collection_exists(collection):
            return False
        condition = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        return self.client.count(collection, count_filter=condition, exact=True).count > 0

    def stats(self, collection: str) -> dict[str, int]:
        info = self.client.get_collection(collection)
        return {
            "point_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "vector_dimension": info.config.params.vectors.size,
            "segment_count": info.segments_count,
        }

    def count(self, collection: str) -> int:
        return self.client.count(collection, exact=True).count
