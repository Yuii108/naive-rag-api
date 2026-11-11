
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = (
    "正式员工每年享有10天带薪年假。"
    "试用期员工暂不享有带薪年假。"
    "员工每天上午9点上班，下午6点下班。"
    "每周五可以申请远程办公。"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=10,
    length_function=len,
    separators=["。", "，", ""],
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"第{i}块：{chunk}")
