
from sentence_transformers import SentenceTransformer

# 加载中文 Embedding 模型
model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

texts = [
    "正式员工每年享有10天带薪年假。",
    "公司员工每年可以休十天带薪假。"
]

# 将文本转成向量
vectors = model.encode(
    texts,
    normalize_embeddings=True
)

print("向量形状:", vectors.shape)
print("第一个向量的前5个数字:", vectors[0][:5])
