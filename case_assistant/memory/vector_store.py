"""向量库 (ChromaDB) - 语义检索"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ChromaDB 兼容性导入：try-except 降级处理
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None
    embedding_functions = None
    logger.warning("chromadb 未安装，向量库功能不可用，请安装 chromadb>=0.4.0")

# 预定义 Collection 名称
COLLECTION_DOSSIERS = "dossiers"                  # 案件卷宗（按案件分块）
COLLECTION_PERSON_PROFILES = "person_profiles"   # 人员历史画像
COLLECTION_CASE_KNOWLEDGE = "case_knowledge"     # 案件类型相关知识

DEFAULT_COLLECTIONS = [
    COLLECTION_DOSSIERS,
    COLLECTION_PERSON_PROFILES,
    COLLECTION_CASE_KNOWLEDGE,
]


def _build_embedding_function():
    """构建 embedding function，按可用性降级处理"""
    if not CHROMA_AVAILABLE:
        return None
    # 优先尝试 chromadb 默认 embedding（基于 ONNX 的 MiniLM，通常开箱即用）
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        ef = DefaultEmbeddingFunction()
        # 触发一次嵌入以验证可用
        ef(["test"])
        return ef
    except Exception as e:
        logger.warning("chromadb DefaultEmbeddingFunction 不可用: %s", e)
    # 降级：尝试 sentence-transformers
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        ef(["test"])
        return ef
    except Exception as e:
        logger.warning("sentence-transformers 不可用: %s", e)
    # 全部不可用，返回 None，由 chromadb 自动选择默认 embedding
    logger.warning("所有显式 embedding function 均不可用，将交由 chromadb 自动选择")
    return None


class VectorStore:
    """ChromaDB 向量库封装"""

    def __init__(self, persist_dir: str):
        """初始化 ChromaDB 客户端，创建或加载持久化目录"""
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb 未安装，无法初始化向量库")

        self.persist_dir = persist_dir
        try:
            os.makedirs(persist_dir, exist_ok=True)
        except Exception as e:
            logger.warning("创建持久化目录失败 %s: %s", persist_dir, e)

        self._embedding_function = _build_embedding_function()

        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            logger.error("初始化 ChromaDB PersistentClient 失败: %s", e)
            raise

        # 初始化默认 collections
        self._collections: Dict[str, Any] = {}
        for name in DEFAULT_COLLECTIONS:
            try:
                self._collections[name] = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self._embedding_function,
                )
            except Exception as e:
                logger.warning("初始化 collection %s 失败: %s", name, e)

    def _get_collection(self, collection_name: str):
        """获取 collection，不存在则创建"""
        if collection_name in self._collections:
            return self._collections[collection_name]
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_function,
            )
            self._collections[collection_name] = collection
            return collection
        except Exception as e:
            logger.error("获取/创建 collection %s 失败: %s", collection_name, e)
            raise

    def add_documents(self, collection_name: str, documents: List[str],
                      metadatas: List[Dict], ids: List[str]) -> None:
        """向指定 collection 添加文档"""
        if not documents:
            return
        # 参数校验
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError(
                "documents(%d)、metadatas(%d)、ids(%d) 长度不一致"
                % (len(documents), len(metadatas), len(ids))
            )
        collection = self._get_collection(collection_name)
        try:
            # chromadb 仅支持 str/int/float/bool/None 类型的 metadata 值
            safe_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            collection.add(
                documents=documents,
                metadatas=safe_metadatas,
                ids=ids,
            )
        except Exception as e:
            logger.error("向 collection %s 添加文档失败: %s", collection_name, e)
            raise

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict]) -> Dict:
        """规范化 metadata 值，chromadb 仅支持基本类型"""
        if not metadata:
            return {}
        safe: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, (list, dict, tuple)):
                # 复杂类型序列化为字符串
                try:
                    safe[k] = json.dumps(v, ensure_ascii=False, default=str)
                except Exception:
                    safe[k] = str(v)
            else:
                # 兜底：日期等对象转为字符串
                safe[k] = str(v)
        return safe

    def search(self, collection_name: str, query: str,
               n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """语义检索，支持元数据过滤"""
        collection = self._get_collection(collection_name)
        try:
            kwargs: Dict[str, Any] = {
                "query_texts": [query],
                "n_results": n_results,
            }
            if where:
                kwargs["where"] = where
            results = collection.query(**kwargs)
        except Exception as e:
            logger.error("检索 collection %s 失败: %s", collection_name, e)
            return []

        return self._format_query_results(results)

    @staticmethod
    def _format_query_results(results: Dict) -> List[Dict[str, Any]]:
        """统一格式化 query 返回结果（chromadb 不同版本字段结构基本一致）"""
        if not results or not results.get("ids"):
            return []
        # query 基于 query_texts 列表，每条 query 对应一组结果，这里取第 0 条
        ids_list = results.get("ids", [[]])
        documents_list = results.get("documents", [[]])
        metadatas_list = results.get("metadatas", [[]])
        distances_list = results.get("distances", [[]])

        ids = ids_list[0] if ids_list else []
        documents = documents_list[0] if documents_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        distances = distances_list[0] if distances_list else []

        formatted: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            formatted.append({
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None,
            })
        return formatted

    def get_collection_names(self) -> List[str]:
        """获取所有 collection 名称"""
        try:
            collections = self.client.list_collections()
        except Exception as e:
            logger.warning("获取 collection 列表失败: %s", e)
            return list(self._collections.keys())

        names: List[str] = []
        for c in collections:
            # 兼容不同 chromadb 版本：可能返回字符串或 Collection 对象
            if isinstance(c, str):
                names.append(c)
            else:
                names.append(getattr(c, "name", str(c)))
        # 合并缓存中的名称（保证默认 collection 一定出现）
        for name in self._collections.keys():
            if name not in names:
                names.append(name)
        return names


# 全局单例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取向量库全局单例"""
    global _vector_store
    if _vector_store is None:
        # 延迟导入以避免循环依赖，同时兼容绝对/相对导入
        try:
            from case_assistant.config import CHROMA_PERSIST_DIR
        except ImportError:
            from ..config import CHROMA_PERSIST_DIR
        _vector_store = VectorStore(CHROMA_PERSIST_DIR)
    return _vector_store
