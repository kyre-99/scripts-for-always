from tqdm import tqdm
from datasets import concatenate_datasets
from vdb import construct_vdb, VectorDatabaseData, compute_mdhash_id, VdbConfig, ChromaVectorDatabase
from datasets import load_dataset
from embed import EmbeddingConfig, Embedding
import sys
sys.path.append('/data/wangzehua/Projects/Agent_Tools/rag')
data = load_dataset(
    '/data/wangzehua/Projects/verl-main/datasets/data/2wikimultihopqa/all')
train_data = data['train']  # 取1000
test_data = data['test']  # 取200
data = load_dataset(
    '/data/wangzehua/Projects/verl-main/datasets/data/2wikimultihopqa/split')
train_data2 = data['train'].select(range(1000))  # 取1000
test_data2 = data['test'].select(range(200))  # 取200

# 合并训练集和测试集
combined_data = concatenate_datasets([train_data, test_data])

# 查看合并后的数据集大小
print(f"合并前训练集大小: {len(train_data)}")
print(f"合并前测试集大小: {len(test_data)}")
print(f"合并后数据集大小: {len(combined_data)}")


def construct_ids_docs(data):
    ids = []
    docs = []
    for d in tqdm(data):
        titles = d['metadata']['context']['title']
        content = d['metadata']['context']['content']
        assert len(titles) == len(content), "titles is not equal contents"
        for t, c in zip(titles, content):
            text = t+"\n"+" ".join(c)
            text_id = compute_mdhash_id(text, prefix="2wikimultihopqa-")
            if text_id in ids:
                # print('repeat',t)
                continue
            docs.append(text)
            ids.append(text_id)
    return ids, docs


ids, docs = construct_ids_docs(combined_data)

# 配置 embedded
embed_config = EmbeddingConfig('', '', '')
embed_func = Embedding(embed_config)
# 配置vdb
vdb_config = VdbConfig(
    vdb_path='/data/wangzehua/Projects/verl-main/datasets/data/2wikimultihopqa/all/db')
vdb = ChromaVectorDatabase(vdb_config, embed_func=embed_func)
for i in tqdm(range(0, len(ids), 5)):
    if i+5 > len(ids):
        new_data = {
            "documents": docs[i:],
            "ids": ids[i:],
        }
    else:
        new_data = {
            "documents": docs[i:i+5],
            "ids": ids[i:i+5],
        }
    construct_vdb(new_data, vdb)
