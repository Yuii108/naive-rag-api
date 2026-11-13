
"""执行原项目文档的 Milvus 入库与同步"""

from dotenv import load_dotenv

from indexer import prepare_chunks

from milvus_store import (
    connect_milvus,
    sync_chunks,
    COLLECTION_NAME,
)


def main():

    # 1. 加载环境变量
    load_dotenv()

    # 2. 加载、分割原项目文档
    chunks = prepare_chunks()

    print(f"本地分块数：{len(chunks)}")

    # 3. 连接 Milvus
    client = connect_milvus()

    try:
        # 4. 同步数据
        stats = sync_chunks(client, chunks)

        print("Collection:", COLLECTION_NAME)
        print("新增：", stats["new"])
        print("更新：", stats["updated"])
        print("未变化：", stats["unchanged"])
        print("删除旧记录：", stats["deleted"])
        print("本次文档块总数：", stats["total"])

    finally:
        # 5. 关闭客户端
        client.close()


if __name__ == "__main__":
    main()
