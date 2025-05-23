from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model import RAGnInputModel,QueryRequest
from rag.utils import RetrievalSystem
from rag.vdb import ChromaVectorDatabase,VdbConfig
from rag.embed import Embedding, EmbeddingConfig
## CMD uvicorn app:app --reload --port 8080  --host 0.0.0.0
app = FastAPI()
k = 16
rrf_k = 100
retrieval_system = RetrievalSystem(retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus/", HNSW=False, cache=True)

## 配置 embedded
embed_config= EmbeddingConfig('','','')
embed_func = Embedding(embed_config)
## 配置vdb
vdb_config = VdbConfig(vdb_path="/data/wangzehua/Projects/Scripts/db/vdb/chunk")
vdb = ChromaVectorDatabase(vdb_config,embed_func=embed_func)
print("Retrieval system initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/rag")
async def rag(request: RAGnInputModel) :
    retrieved_snippets, scores = retrieval_system.retrieve(request.query, k=k, rrf_k=rrf_k)
    contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
    try:
        return {"contexts": contexts,
                "snippets": retrieved_snippets,
                "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    
    # Perform batch retrieval
    
    results, scores = retrieval_system.batch_search(
        query_list=request.queries,
        k=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


@app.post("/retrieve_htopot")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    Return format:
    {
      "result": [
        [
         {"document": {"title": "Document 1", "content": "Content of document 1"}, "score": 0.9},
         {"document": {"title": "Document 2", "content": "Content of document 2"}, "score": 0.8}
        ],
        [
            {"document": {"title": "Document 3", "content": "Content of document 3"}, "score": 0.85},
            {"document": {"title": "Document 4", "content": "Content of document 4"}, "score": 0.75}
        ]
        ]
    }
    """
    
    # Perform batch retrieval
    try:
        results = vdb.query(
            query_texts=request.queries,
            topk=request.topk,
            include=["documents"])
        # return results
        documents = results["documents"]
        # scores = results["distances"]
        # Format response
        resp = []
        for i, single_result in enumerate(documents):
            combined = []
            for doc in single_result:
                title = doc.split("\n")[0]
                content = "\n".join(doc.split("\n")[1:])
                combined.append({"document": {"title": title, "content": content}, "score": 0})
            resp.append(combined)
    except Exception as e:
        print(f"Error during retrieval: {request.queries}")
       
    return {"result": resp}