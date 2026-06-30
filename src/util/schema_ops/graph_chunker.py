import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from typing import List, Dict
import numpy as np

from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.embeddings.encoder import embed_texts


class SegmentData:
    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.facts: List[AtomicFact] = []


def build_fused_graph(
    segments: List[SegmentData], k: int, embeddings: np.ndarray
) -> nx.Graph:
    G = nx.Graph()
    for i in range(len(segments)):
        G.add_node(i)

    # 1. Positional Edges
    # Sort segments by start_char
    sorted_indices = sorted(range(len(segments)), key=lambda i: segments[i].start_char)
    for idx in range(len(sorted_indices) - 1):
        i = sorted_indices[idx]
        j = sorted_indices[idx + 1]

        # Calculate gap
        gap = max(0, segments[j].start_char - segments[i].end_char) + 1
        weight = 1.0 / gap
        G.add_edge(i, j, weight=weight)

    # 2. Semantic Edges (Mutual k-NN)
    if len(segments) > 1:
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, -1.0)  # Ignore self

        # Get top-k for each node
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        top_k_sets = [set(indices) for indices in top_k_indices]

        for i in range(len(segments)):
            for j in top_k_sets[i]:
                if i in top_k_sets[j]:
                    # Mutual k-NN!
                    sim_weight = sim_matrix[i, j]
                    if sim_weight > 0:
                        if G.has_edge(i, j):
                            G[i][j]["weight"] += sim_weight
                        else:
                            G.add_edge(i, j, weight=sim_weight)
    return G


def run_graph_chunker(facts: List[AtomicFact]) -> ChunkedPlan:
    # 1. Group facts into segments
    segment_map: Dict[str, SegmentData] = {}
    standalone_facts: List[AtomicFact] = []

    for f in facts:
        if not f.segment_text:
            standalone_facts.append(f)
            continue

        if f.segment_text not in segment_map:
            segment_map[f.segment_text] = SegmentData(
                f.segment_text, f.start_char, f.end_char
            )
        segment_map[f.segment_text].facts.append(f)

    segments = list(segment_map.values())

    if len(segments) <= 1:
        # Fallback if there's no meaningful segmentation
        return ChunkedPlan(core_modeling_facts=facts, chunks=[facts])

    # 2. Embed segments
    texts = [s.text for s in segments]
    embeddings = embed_texts(texts)

    # 3. Dynamic Plateau Search for k
    K_MAX = min(5, len(segments) - 1)
    if K_MAX < 1:
        K_MAX = 1

    partitions = []
    graphs = []
    for k in range(1, K_MAX + 1):
        G = build_fused_graph(segments, k, embeddings)
        graphs.append(G)

        # Louvain Modularity Maximization
        communities = louvain_communities(G, weight="weight", seed=42)

        # Convert to label array for ARI calculation
        labels = np.zeros(len(segments), dtype=int)
        for c_idx, comm in enumerate(communities):
            for node in comm:
                labels[node] = c_idx
        partitions.append((k, communities, labels))

    # Find plateau
    best_k_idx = 0
    if len(partitions) > 1:
        aris = []
        for i in range(len(partitions) - 1):
            ari = adjusted_rand_score(partitions[i][2], partitions[i + 1][2])
            aris.append(ari)

        # Select highest ARI indicating stability plateau
        best_k_idx = np.argmax(aris) + 1  # biases towards higher k in plateau

    chosen_k, chosen_communities, _ = partitions[best_k_idx]
    chosen_G = graphs[best_k_idx]

    # Map each node to its community index, and identify BRIDGE nodes: segments whose
    # neighbors span >= 2 communities. Only bridges are duplicated into adjacent chunks.
    node_comm = {
        node: idx for idx, comm in enumerate(chosen_communities) for node in comm
    }
    bridge_nodes = set()
    for node in chosen_G.nodes():
        neighbor_comms = {node_comm[n] for n in chosen_G.neighbors(node)}
        neighbor_comms.add(node_comm[node])
        if len(neighbor_comms) >= 2:
            bridge_nodes.add(node)

    # 4. Form chunks and apply a BOUNDED halo extension.
    final_chunks: List[List[AtomicFact]] = []
    for comm in chosen_communities:
        chunk_facts = []
        # Core facts
        for node in comm:
            chunk_facts.extend(segments[node].facts)

        # Bounded halo: pull in ONLY boundary neighbors that are genuine bridges (connect
        # >= 2 communities), not every adjacent segment. The positional chain otherwise
        # links all consecutive segments and bloats each chunk toward the full fact set.
        boundary_bridges = set()
        for node in comm:
            for neighbor in chosen_G.neighbors(node):
                if neighbor not in comm and neighbor in bridge_nodes:
                    boundary_bridges.add(neighbor)

        for b_node in boundary_bridges:
            chunk_facts.extend(segments[b_node].facts)

        # Include standalone facts in every chunk (metadata/context)
        chunk_facts.extend(standalone_facts)

        # Deduplicate facts in chunk (just in case)
        unique_facts = {f.id: f for f in chunk_facts}
        final_chunks.append(list(unique_facts.values()))

    return ChunkedPlan(core_modeling_facts=facts, chunks=final_chunks)
