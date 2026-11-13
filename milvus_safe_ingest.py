
"""Day 6: 单机维护窗口内的安全同步入口。"""

import argparse
import fcntl
import json
from pathlib import Path

from dotenv import load_dotenv
from indexer import prepare_chunks

from milvus_store import (
    COLLECTION_NAME,
    connect_milvus,
    make_hash,
    make_id,
    read_existing_hashes,
    sync_chunks,
)


ROOT = Path(__file__).resolve().parent

LOCK_FILE = ROOT / ".milvus_sync.lock"
MARKER_FILE = ROOT / ".milvus_sync_incomplete"


# 1. 计算更新计划，不修改数据库
def build_plan(chunks, old_hashes):

    if not chunks:
        raise ValueError(
            "源文档没有生成任何 Chunk，拒绝同步"
        )

    expected = {}

    for chunk in chunks:
        doc_id = make_id(chunk)

        # 防止两个不同 Chunk 生成相同主键
        if doc_id in expected:
            raise ValueError(
                f"不同 Chunk 生成重复 ID：{doc_id}"
            )

        expected[doc_id] = make_hash(chunk)

    # 新增记录
    added = set(expected) - set(old_hashes)

    # 已删除的旧记录
    removed = set(old_hashes) - set(expected)

    # ID 相同但内容摘要不同
    changed = {
        doc_id
        for doc_id in set(expected) & set(old_hashes)
        if expected[doc_id] != old_hashes[doc_id]
    }

    unchanged = (
        len(expected)
        - len(added)
        - len(changed)
    )

    plan = {
        "new": len(added),
        "updated": len(changed),
        "unchanged": unchanged,
        "deleted": len(removed),
        "total": len(expected),
    }

    return expected, plan


# 2. 主程序
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="确认执行更新",
    )

    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="允许删除旧文档块",
    )

    parser.add_argument(
        "--simulate-failure",
        action="store_true",
        help="写入前模拟故障",
    )

    args = parser.parse_args()

    if args.simulate_failure and not args.apply:
        parser.error(
            "--simulate-failure 必须与 --apply 一起使用"
        )

    load_dotenv()

    # 3. 获取本机独占锁，防止本脚本同时运行两份
    with LOCK_FILE.open("a+") as lock:

        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "已有另一个安全入库进程在运行"
            ) from exc

        # 4. 加载本地文档块
        chunks = prepare_chunks()

        client = connect_milvus()

        try:
            # 5. 查询数据库现有摘要
            if client.has_collection(
                collection_name=COLLECTION_NAME
            ):
                client.load_collection(
                    collection_name=COLLECTION_NAME
                )

                old = read_existing_hashes(client)

            else:
                old = {}

            # 6. 只计算计划，不生成 Embedding
            expected, plan = build_plan(
                chunks,
                old,
            )

            print("Collection:", COLLECTION_NAME)
            print(
                "同步计划:",
                json.dumps(
                    plan,
                    ensure_ascii=False,
                ),
            )

            if MARKER_FILE.exists():
                print(
                    "警告：上一次同步尚未验证完成。"
                )

            # 7. 默认只预览
            if not args.apply:
                print("DRY RUN：没有修改数据库")
                return

            # 8. 默认不允许删除旧块
            if plan["deleted"] and not args.allow_delete:
                raise RuntimeError(
                    "本次将删除旧块。"
                    "检查源文件后，再加 "
                    "--allow-delete 确认"
                )

            # 9. 开始同步前写入故障标记
            MARKER_FILE.write_text(
                json.dumps(
                    plan,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            try:
                # 10. 用于故障恢复实验
                if args.simulate_failure:
                    raise RuntimeError(
                        "模拟故障：未执行任何数据库写入"
                    )

                # 11. 复用 Day 3 的同步功能
                result = sync_chunks(
                    client,
                    chunks,
                )

                # 12. 同步后重新读取数据库
                actual = read_existing_hashes(
                    client
                )

                # 13. 校验所有 ID 和摘要
                if actual != expected:
                    raise RuntimeError(
                        "同步后 ID / 内容摘要校验失败"
                    )

            except Exception:
                print(
                    "同步未完成：保留 "
                    ".milvus_sync_incomplete，"
                    "拒绝启动在线服务"
                )
                raise

            else:
                # 14. 完成验证，解除故障标记
                MARKER_FILE.unlink()

                print("同步成功：", result)
                print(
                    "已验证数据库与源文档一致"
                )

        finally:
            client.close()


if __name__ == "__main__":
    main()
