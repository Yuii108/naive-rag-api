
"""Milvus 持久化存储模块"""
from indexer import MODEL_NAME
import hashlib
import json
import os

from pymilvus import DataType, MilvusClient


COLLECTION_NAME = "rag_employee_v1"
VECTOR_DIM = 512


# 1. 连接 Milvus
def connect_milvus():
    uri = os.getenv(
        "MILVUS_URI",
        "http://localhost:19530"
    )

    token = os.getenv("MILVUS_TOKEN")

    if token:
        return MilvusClient(uri=uri, token=token)

    return MilvusClient(uri=uri)


# 2. 给文档块生成稳定主键
def make_id(chunk):
    meta = chunk.metadata

    identity = json.dumps(
        [
            meta["source"],
            meta.get("page"),
            meta.get("start_index", 0),
        ],
        ensure_ascii=False,
    )

    raw = hashlib.sha256(
        identity.encode("utf-8")
    ).digest()

    # 转成 Milvus INT64 支持的非负整数
    return int.from_bytes(
        raw[:8], "big"
    ) & ((1 << 63) - 1)


# 3. 生成内容摘要
def make_hash(chunk):
    meta = chunk.metadata

    data = {
        "text": chunk.page_content,
        "source": meta["source"],
        "page": meta.get("page"),
        "chunk_id": meta["chunk_id"],
        "file_type": meta.get("file_type", ""),
        "embedding_model": MODEL_NAME,
        "index_config_version": os.getenv(
            "RAG_INDEX_VERSION", "v1"
        ),
    }

    content = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# 4. 确保 Collection 存在
def ensure_collection(client):

    if not client.has_collection(
        collection_name=COLLECTION_NAME
    ):

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            "id", DataType.INT64, is_primary=True
        )

        schema.add_field(
            "text", DataType.VARCHAR, max_length=8192
        )

        schema.add_field(
            "source", DataType.VARCHAR, max_length=256
        )

        schema.add_field(
            "file_type", DataType.VARCHAR, max_length=16
        )

        schema.add_field("page", DataType.INT64)
        schema.add_field("start_index", DataType.INT64)
        schema.add_field("chunk_id", DataType.INT64)

        schema.add_field(
            "content_hash",
            DataType.VARCHAR,
            max_length=64,
        )

        schema.add_field(
            "vector",
            DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
        )

        indexes = client.prepare_index_params()

        indexes.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=indexes,
        )

        print("已创建 Collection:", COLLECTION_NAME)

    else:
        # 防止误用结构不兼容的旧 Collection
        desc = client.describe_collection(
            collection_name=COLLECTION_NAME
        )

        names = {
            field["name"] for field in desc["fields"]
        }

        expected = {
            "id", "text", "source", "file_type",
            "page", "start_index", "chunk_id",
            "content_hash", "vector",
        }

        if not expected.issubset(names):
            raise ValueError(
                "已有 Collection 的 Schema 不匹配"
            )

        vector_field = next(
            field for field in desc["fields"]
            if field["name"] == "vector"
        )

        dim = int(
            vector_field.get("params", {}).get("dim", -1)
        )

        if dim != VECTOR_DIM:
            raise ValueError(
                "已有 Collection 的向量维度不是 512"
            )

    client.load_collection(
        collection_name=COLLECTION_NAME,
        replica_number=1,
    )


# 5. 读取已有记录的 ID 和摘要
def read_existing_hashes(client):

    iterator = client.query_iterator(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["id", "content_hash"],
        batch_size=500,
        consistency_level="Strong",
    )

    existing = {}

    try:
        while True:
            rows = iterator.next()

            if not rows:
                break

            for row in rows:
                existing[int(row["id"])] = row["content_hash"]

    finally:
        iterator.close()

    return existing


# 6. 同步文档到 Milvus
def sync_chunks(client, chunks):

    if not chunks:
        raise ValueError(
            "文档块为空，为安全起见不清空数据库"
        )

    ensure_collection(client)

    old = read_existing_hashes(client)

    current_ids = set()
    pending = []

    inserted = 0
    updated = 0
    unchanged = 0

    # 第一步：比较新旧记录
    for chunk in chunks:

        doc_id = make_id(chunk)

        if doc_id in current_ids:
            raise ValueError(
                f"两个文档块出现相同 ID：{doc_id}"
            )

        current_ids.add(doc_id)
        digest = make_hash(chunk)

        if old.get(doc_id) == digest:
            unchanged += 1
            continue

        pending.append(
            (doc_id, chunk, digest, doc_id in old)
        )

    # 第二步：只为新增或变化的数据生成向量
    if pending:

        from indexer import load_embedding_model

        model = load_embedding_model()

        texts = [
            chunk.page_content
            for _, chunk, _, _ in pending
        ]

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if vectors.shape != (len(pending), VECTOR_DIM):
            raise ValueError(
                f"向量维度不匹配：{vectors.shape}"
            )

        new_records = []
        changed_records = []

        for (
            doc_id, chunk, digest, existed
        ), vector in zip(pending, vectors):

            meta = chunk.metadata

            record = {
                "id": doc_id,
                "text": chunk.page_content,
                "source": meta["source"],
                "file_type": meta.get("file_type", ""),
                "page": (
                    int(meta["page"])
                    if meta.get("page") is not None
                    else -1
                ),
                "start_index": int(
                    meta.get("start_index", 0)
                ),
                "chunk_id": int(meta["chunk_id"]),
                "content_hash": digest,
                "vector": vector.tolist(),
            }

            if existed:
                changed_records.append(record)
            else:
                new_records.append(record)

        # 第三步：插入新数据
        if new_records:
            client.insert(
                collection_name=COLLECTION_NAME,
                data=new_records,
            )

            inserted = len(new_records)

        # 第四步：更新变化的数据
        if changed_records:
            client.upsert(
                collection_name=COLLECTION_NAME,
                data=changed_records,
            )

            updated = len(changed_records)

    # 第五步：删除已经不在源文档中的旧块
    stale_ids = sorted(
        set(old) - current_ids
    )

    for start in range(0, len(stale_ids), 500):
        client.delete(
            collection_name=COLLECTION_NAME,
            ids=stale_ids[start:start + 500],
        )

    return {
        "new": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "deleted": len(stale_ids),
        "total": len(chunks),
    }
