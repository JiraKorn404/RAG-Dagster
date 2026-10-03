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

    def documents(self, collection: str) -> list[dict]:
        """The documents in a collection: for each, its id, number of points, file name and when it was
        ingested. Counts come from one facet call on the indexed `doc_id`; the name and time from one
        point of the document. A collection that was not made by this app may have no such index; it is
        then read by scrolling through its points."""
        try:
            hits = self.client.facet(collection, key="doc_id", limit=1000, exact=True).hits
        except Exception:  # noqa: BLE001  (no payload index on doc_id: count by scrolling instead)
            return self._documents_by_scrolling(collection)
        docs = []
        for hit in hits:
            condition = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=hit.value))])
            points, _ = self.client.scroll(
                collection, scroll_filter=condition, limit=1, with_payload=["source_file", "ingested_at"]
            )
            payload = points[0].payload if points else {}
            docs.append(
                {
                    "doc_id": hit.value,
                    "points": hit.count,
                    "source_file": payload.get("source_file"),
                    "ingested_at": payload.get("ingested_at"),
                }
            )
        return sorted(docs, key=lambda d: (d["source_file"] or "", d["doc_id"]))

    def _documents_by_scrolling(self, collection: str) -> list[dict]:
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection, limit=1000, offset=offset, with_payload=["doc_id", "source_file", "ingested_at"]
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id is None:
                    continue
                doc = docs.setdefault(
                    doc_id,
                    {"doc_id": doc_id, "points": 0, "source_file": payload.get("source_file"), "ingested_at": payload.get("ingested_at")},
                )
                doc["points"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: (d["source_file"] or "", d["doc_id"]))

    def count_document(self, collection: str, doc_id: str) -> int:
        condition = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        return self.client.count(collection, count_filter=condition, exact=True).count

    def delete_collection(self, collection: str) -> None:
        self.client.delete_collection(collection)

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
