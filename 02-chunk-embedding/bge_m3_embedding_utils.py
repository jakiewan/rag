"""
bge_m3_embedding_utils.py — BGE-M3 单模型产出稠密 + 稀疏向量。

稠密：1024 维 float 列表（Milvus FLOAT_VECTOR）
稀疏：{token_id: weight} 字典（Milvus SPARSE_FLOAT_VECTOR 直接吃），已 L2 归一化

配置从环境变量读：BGE_M3_MODEL_PATH / BGE_M3_USE_FP16（详见 .env.example）。
"""
import logging
import math
import os
from typing import Dict, List, Optional

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bge_m3")

DENSE_DIM = 1024
DEFAULT_MODEL_PATH = (
    "/Volumes/SSD-工作/models/caches/hf-cache/hub/"
    "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)

_model: Optional[BGEM3EmbeddingFunction] = None


def _get_model() -> BGEM3EmbeddingFunction:
    global _model
    if _model is None:
        path = os.getenv("BGE_M3_MODEL_PATH") or DEFAULT_MODEL_PATH
        use_fp16 = os.getenv("BGE_M3_USE_FP16", "true").lower() in ("1", "true", "yes")
        _model = BGEM3EmbeddingFunction(model_name=path, use_fp16=use_fp16)
        logger.info(f"BGE-M3 加载: {path} fp16={use_fp16}")
    return _model


def _l2_normalize(d: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(w * w for w in d.values()))
    return d if norm == 0 else {k: v / norm for k, v in d.items()}


def generate_hybrid_embedding(docs: List[str]) -> Dict[str, List]:
    """生成稠密 + 稀疏（已 L2 归一化）向量。

    Returns:
        {"dense": [[float]*1024, ...], "sparse": [{token_id: weight}, ...]}
    """
    raw = _get_model().encode_documents(docs)
    csr = raw["sparse"]
    sparse = []
    for i in range(len(docs)):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        d = dict(zip(csr.indices[s:e].tolist(), csr.data[s:e].tolist()))
        sparse.append(_l2_normalize(d))
    dense = [v.tolist() for v in raw["dense"]]

    assert len(dense) == len(sparse) == len(docs), "稠密/稀疏/输入 数量不一致"
    assert all(len(v) == DENSE_DIM for v in dense), f"稠密维度不是 {DENSE_DIM}"
    norms = [math.sqrt(sum(w * w for w in s.values())) for s in sparse]
    logger.info(f"嵌入完成 n={len(docs)} dim={DENSE_DIM} sparse_norm[{min(norms):.3f}, {max(norms):.3f}]")
    return {"dense": dense, "sparse": sparse}


def to_milvus_sparse(sparse: Dict[int, float]) -> Dict[int, float]:
    """Milvus SPARSE_FLOAT_VECTOR 直接吃这个 dict；这里仅做断言便于上层调用。"""
    assert all(isinstance(k, int) and k >= 0 for k in sparse), "sparse key 必须是非负 int"
    assert all(isinstance(v, float) for v in sparse.values()), "sparse value 必须是 float"
    return sparse


if __name__ == "__main__":
    r = generate_hybrid_embedding(["hello world", "你好世界"])
    print(f"稠密维度: {len(r['dense'][0])} | 稀疏非零项: {[len(s) for s in r['sparse']]}")
    print(f"sparse[0] 前 3 项: {list(r['sparse'][0].items())[:3]}")