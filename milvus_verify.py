
"""直接从 Milvus 读取持久化的文档数据"""

from dotenv import load_dotenv

from milvus_store import (
    connect_milvus,
    COLLECTION_NAME,
)


def main():
    load_dotenv()

    client = connect_milvus()

    try:
        if not client.has_collection(
            collection_name=COLLECTION_NAME
        ):
            raise RuntimeError(
                "Collection 不存在，请先运行 milvus_ingest.py"
            )

        client.load_collection(
            collection_name=COLLECTION_NAME,
            replica_number=1,
        )

        # 统计记录数量
        count = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )

        # 直接读取前三条记录
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=[
                "id",
                "text",
                "source",
                "chunk_id",
            ],
            limit=3,
            consistency_level="Strong",
        )

        print("Collection:", COLLECTION_NAME)
        print("数据库记录数：", count)

        for row in rows:
            print(
                f"[块 {row['chunk_id']}] "
                f"{row['source']}: "
                f"{row['text'][:80]}"
            )

        if not rows:
            raise AssertionError("查询结果为空")

    finally:
        client.close()


if __name__ == "__main__":
    main()
