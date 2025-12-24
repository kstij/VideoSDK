import os
from typing import List, Tuple
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
class RAGPipeline:
    def __init__(self, docs_folder: str, openai_api_key: str, similarity_threshold: float = 0.75, k: int = 3):
        self.docs_folder = docs_folder
        self.similarity_threshold = similarity_threshold
        self.k = k
        self.openai_embedder = OpenAIEmbeddings(api_key=openai_api_key)
        self.index = None
        self.docs = []
        self.texts = []
        self.doc_text_map = []
        self._ingest_docs()
    
    def _ingest_docs(self):
        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        cur_idx = 0
        for fname in sorted(os.listdir(self.docs_folder)):
            path = os.path.join(self.docs_folder, fname)
            if os.path.isfile(path) and fname.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
                    for chunk in splitter.split_text(doc_text):
                        self.texts.append(chunk)
                        self.doc_text_map.append((cur_idx, fname, chunk))
                        cur_idx += 1
        if self.texts:
            vectors = self.openai_embedder.embed_documents(self.texts)
            self.index = faiss.IndexFlatL2(len(vectors[0]))
            self.index.add(np.array(vectors).astype('float32'))
    
    def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
        if not self.index:
            return [], []
        query_vec = self.openai_embedder.embed_query(query)
        D, I = self.index.search(np.array([query_vec]).astype('float32'), self.k)
        scores = 1 - D[0] / 2
        results = []
        result_scores = []
        for idx, sc in zip(I[0], scores):
            if idx < 0 or sc < self.similarity_threshold:
                continue
            _, fname, text = self.doc_text_map[idx]
            results.append(text)
            result_scores.append(sc)
        return results, result_scores

