import os

from dotenv import load_dotenv
from pymilvus import MilvusClient


# 1. 读取环境变量
load_dotenv()

# 2. Milvus 服务器地址
uri = os.getenv(
    "MILVUS_URI",
    "http://localhost:19530"
)

# 3. 创建客户端连接
client = MilvusClient(uri=uri)

# 4. 查询已有的 Collection
collections = client.list_collections()

print("Milvus 连接成功！")
print("当前 Collections:", collections)

# 5. 关闭客户端
client.close()