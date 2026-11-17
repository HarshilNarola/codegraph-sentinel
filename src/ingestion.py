import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List

from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = [".py", ".js", ".ts", ".cpp", ".c", ".java", ".go", ".rs"]


def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repository(repo_url: str, target_dir: str = "./workspace/repo") -> str:
    """Clone a public Git repository into a clean target directory."""
    repo_url = repo_url.strip()
    if not repo_url.startswith(("https://", "http://", "git@")):
        raise ValueError("Repository URL must be an HTTPS, HTTP, or SSH Git URL.")

    target_path = Path(target_dir)
    if target_path.exists():
        shutil.rmtree(target_path, onerror=_remove_readonly)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(repo_url, str(target_path), depth=1)
    return str(target_path)


def parse_codebase_ast(repo_dir: str) -> List[Document]:
    """Parse supported source files into AST-aware chunks with file metadata."""
    loader = GenericLoader.from_filesystem(
        repo_dir,
        glob="**/*",
        suffixes=SUPPORTED_SUFFIXES,
        exclude=["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**"],
        parser=LanguageParser(parser_threshold=10),
    )
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    processed_docs: List[Document] = []

    for doc in raw_docs:
        source = doc.metadata.get("source", "unknown")
        rel_path = os.path.relpath(source, repo_dir)
        doc.metadata["file_path"] = rel_path
        doc.metadata["language"] = doc.metadata.get(
            "language", Path(rel_path).suffix.lstrip(".") or "text"
        )
        if len(doc.page_content) > 2000:
            processed_docs.extend(splitter.split_documents([doc]))
        elif doc.page_content.strip():
            processed_docs.append(doc)
    return processed_docs


class HybridCodeRetriever:
    """Fuse BM25 and local HuggingFace/Chroma results with reciprocal rank fusion."""

    def __init__(self, docs: List[Document], persist_dir: str = "./workspace/chroma_db"):
        if not docs:
            raise ValueError("Cannot build a retriever without source documents.")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, onerror=_remove_readonly)

        self.docs = docs
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma.from_documents(
            documents=docs, embedding=self.embeddings, persist_directory=persist_dir
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 6

    @staticmethod
    def _key(doc: Document) -> str:
        return f"{doc.metadata.get('file_path', 'unknown')}::{hash(doc.page_content)}"

    def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense_hits = self.dense_retriever.invoke(query)
        sparse_hits = self.bm25_retriever.invoke(query)
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_hits + sparse_hits):
            key = self._key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (
                rrf_k + (rank % 6) + 1
            )

        results = []
        for key in sorted(scores, key=scores.get, reverse=True)[:top_k]:
            doc = doc_map[key]
            results.append(
                {
                    "file_path": doc.metadata.get("file_path", "unknown"),
                    "language": doc.metadata.get("language", "text"),
                    "content": doc.page_content,
                    "rrf_score": round(scores[key], 4),
                }
            )
        return results
