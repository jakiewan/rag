from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3_model = None
def get_bge_m3_embedding_model():
    '''获取bge_m3的embedding模型'''
    global bge_m3_model
    if bge_m3_model is None:
        try:
            from sentence_transformers import SBERTModel
            bge_m3 = BGEM3EmbeddingFunction(
                model_name="/Volumes/SSD-工作/models/caches/hf-cache/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
                # 本地快照绝对路径（snapshot 哈希是 HF 自动生成的，每次重下都会变）
                use_fp16=True,  # 半精度加载：内存 ~1 GB（fp32 是 2.1 GB）
            )
        except Exception as e:
            print(f"Error loading BGE-M3 model: {e}")
    return bge_m3_model


def generate_hybrid_embedding(docs:List[str]):
    '''Generate hybrid embedding for a list of documents.'''
    #1 获取嵌入模型
    model = get_bge_m3_embedding_model()
    #2 生成稠密 稀疏向量
    documents = model.encode_documents(docs)
    
    #3