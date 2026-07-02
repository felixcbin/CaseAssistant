"""知识图谱 (NetworkX) - 关系推理"""
import logging
import os
from typing import List, Dict, Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

# 节点类型：人员、地点、机构、案件、物品
NODE_TYPES = {"person", "location", "organization", "case", "item"}

# 关系类型：通讯、资金、同行、亲属、同案
RELATION_TYPES = {"communication", "finance", "co-occurrence", "relative", "co-crime"}


class KnowledgeGraph:
    """NetworkX 知识图谱封装

    使用 MultiDiGraph，以 relation 作为边键，从而允许同一对节点之间存在
    多条不同关系的边（如 P001->P002 同时存在"通讯频繁"与"资金往来"）。
    """

    def __init__(self):
        """初始化有向图"""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: str, **attrs) -> None:
        """添加节点（类型：person/location/organization/case/item）"""
        if not node_id:
            raise ValueError("node_id 不能为空")
        if node_type not in NODE_TYPES:
            logger.warning("未知节点类型: %s，仍将添加但建议使用预定义类型", node_type)
        attrs["node_type"] = node_type
        if self.graph.has_node(node_id):
            # 已存在则更新属性（保留旧属性，新属性覆盖同名键）
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, **attrs)

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0, **attrs) -> None:
        """添加边（关系类型：communication/finance/co-occurrence/relative/co-crime）

        以 relation 作为边键，同一对节点可存在多条不同关系的边；
        相同 (source, target, relation) 的重复添加会更新已有边属性。
        """
        if not source or not target:
            raise ValueError("source 和 target 不能为空")
        if relation not in RELATION_TYPES:
            logger.warning("未知关系类型: %s，仍将添加但建议使用预定义类型", relation)
        attrs["relation"] = relation
        attrs["weight"] = float(weight)
        # 自动补充缺失节点，避免后续查询 KeyError
        if not self.graph.has_node(source):
            self.graph.add_node(source, node_type="unknown")
        if not self.graph.has_node(target):
            self.graph.add_node(target, node_type="unknown")
        if self.graph.has_edge(source, target, relation):
            # 相同关系已存在，更新属性
            self.graph.edges[source, target, relation].update(attrs)
        else:
            self.graph.add_edge(source, target, key=relation, **attrs)

    def get_node(self, node_id: str) -> Optional[Dict]:
        """获取节点信息"""
        if not self.graph.has_node(node_id):
            return None
        data = dict(self.graph.nodes[node_id])
        data["id"] = node_id
        return data

    def get_neighbors(self, node_id: str, relation_type: Optional[str] = None) -> List[Dict]:
        """获取节点的邻居（同时考虑出边与入边，每条边返回一个条目）"""
        if not self.graph.has_node(node_id):
            return []
        neighbors: List[Dict] = []
        # 出边邻居：MultiDiGraph 的 out_edges 带 keys 时返回 4 元组
        for _, target, key, data in self.graph.out_edges(node_id, keys=True, data=True):
            if relation_type and key != relation_type:
                continue
            entry = dict(data)
            entry["node_id"] = target
            entry["direction"] = "out"
            entry["node"] = dict(self.graph.nodes[target])
            neighbors.append(entry)
        # 入边邻居
        for source, _, key, data in self.graph.in_edges(node_id, keys=True, data=True):
            if relation_type and key != relation_type:
                continue
            entry = dict(data)
            entry["node_id"] = source
            entry["direction"] = "in"
            entry["node"] = dict(self.graph.nodes[source])
            neighbors.append(entry)
        return neighbors

    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """查找最短路径（基于有向图，忽略边键）"""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return None
        if source == target:
            return [source]
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path
        except nx.NodeNotFound:
            return None
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.warning("查找最短路径失败 %s -> %s: %s", source, target, e)
            return None

    def find_communities(self) -> List[List[str]]:
        """发现社区子群（使用连通分量，将有向图视作无向）"""
        try:
            undirected = self.graph.to_undirected()
            components = nx.connected_components(undirected)
            return [list(c) for c in components]
        except Exception as e:
            logger.warning("社区发现失败: %s", e)
            return []

    def get_centrality(self, top_n: int = 10) -> Dict[str, float]:
        """计算节点中心性（度中心性），返回 top_n 节点"""
        if self.graph.number_of_nodes() == 0:
            return {}
        try:
            centrality = nx.degree_centrality(self.graph)
        except Exception as e:
            logger.warning("度中心性计算失败，降级使用度数: %s", e)
            centrality = {n: float(self.graph.degree(n)) for n in self.graph.nodes()}
        sorted_items = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        if top_n and top_n > 0:
            sorted_items = sorted_items[:top_n]
        return {k: float(v) for k, v in sorted_items}

    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """获取子图，返回 nodes + edges 格式供前端渲染"""
        valid_ids = [n for n in node_ids if self.graph.has_node(n)]
        if not valid_ids:
            return {"nodes": [], "edges": []}
        sub = self.graph.subgraph(valid_ids).copy()
        return self._graph_to_data(sub)

    def to_graphml(self, filepath: str) -> None:
        """持久化为 GraphML 文件"""
        directory = os.path.dirname(filepath)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                logger.warning("创建目录失败 %s: %s", directory, e)
        try:
            nx.write_graphml(self.graph, filepath)
        except Exception as e:
            logger.error("写入 GraphML 失败 %s: %s", filepath, e)
            raise

    def from_graphml(self, filepath: str) -> None:
        """从 GraphML 文件加载"""
        if not os.path.exists(filepath):
            raise FileNotFoundError("GraphML 文件不存在: %s" % filepath)
        try:
            loaded = nx.read_graphml(filepath)
            # read_graphml 可能返回 Graph/DiGraph/MultiGraph/MultiDiGraph，
            # 统一转为 MultiDiGraph 以保持内部数据结构一致
            self.graph = nx.MultiDiGraph(loaded)
        except Exception as e:
            logger.error("读取 GraphML 失败 %s: %s", filepath, e)
            raise

    def get_graph_data(self) -> Dict[str, Any]:
        """返回完整图谱数据（nodes + edges 格式），供前端渲染"""
        return self._graph_to_data(self.graph)

    @staticmethod
    def _graph_to_data(g: nx.Graph) -> Dict[str, Any]:
        """将 NetworkX 图转为前端可渲染的 nodes + edges 结构

        兼容 DiGraph 与 MultiDiGraph：若为多重图则按 keys 展开为多条边。
        """
        nodes: List[Dict[str, Any]] = []
        for node_id, data in g.nodes(data=True):
            node_data = dict(data)
            node_data["id"] = node_id
            # 前端渲染字段：label/group/size，缺失时给默认值
            node_data.setdefault("label", node_data.get("name", node_id))
            node_data.setdefault("group", node_data.get("node_type", "unknown"))
            node_data.setdefault("size", 20)
            nodes.append(node_data)

        edges: List[Dict[str, Any]] = []
        if g.is_multigraph():
            for source, target, _key, data in g.edges(keys=True, data=True):
                edge_data = dict(data)
                edge_data["source"] = source
                edge_data["target"] = target
                edge_data.setdefault("relation", "related")
                edge_data.setdefault("weight", 1.0)
                edges.append(edge_data)
        else:
            for source, target, data in g.edges(data=True):
                edge_data = dict(data)
                edge_data["source"] = source
                edge_data["target"] = target
                edge_data.setdefault("relation", "related")
                edge_data.setdefault("weight", 1.0)
                edges.append(edge_data)
        return {"nodes": nodes, "edges": edges}


# 全局单例
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """获取知识图谱全局单例"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph
