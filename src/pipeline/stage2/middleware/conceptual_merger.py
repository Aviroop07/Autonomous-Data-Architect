import numpy as np
from collections import Counter
from typing import List, Tuple

from src.pipeline.stage2.mapper.conceptual_model import (
    ConceptualModel,
    Entity,
    Relationship,
    Participant,
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.embeddings.encoder import embed_texts
from src.util.algorithms.beta_mixture import (
    compute_merge_probability_matrix,
    compute_flag_posteriors,
)
from src.pipeline.stage2.models.conflicts import ConflictFlag


def merge_all_shards(
    shards: List[ConceptualModel],
    all_facts: List[AtomicFact],
    s_max: int = 300,
    verbose: bool = False,
) -> Tuple[ConceptualModel, List[ConflictFlag]]:
    """
    Takes N shards and merges them using a 2-component Beta Mixture Model.

    Similarities are modelled as a mixture of same-entity (high sim) and
    different-entity (low sim) pairs.  The posterior merge probability
    P(same | sim) drives correlation clustering, and values near 0.5 are
    treated as uncertainty signals for the LLM adjudicator.
    """
    flags = []
    log = print if verbose else lambda *_, **__: None

    # 1. Flatten all entities across shards
    all_entities = []
    shard_map = []  # To track which shard an entity came from (to prevent intra-shard merging)

    for s_idx, shard in enumerate(shards):
        for e in shard.entities:
            all_entities.append(e)
            shard_map.append(s_idx)

    N = len(all_entities)
    log(f"[Merger] {len(shards)} shards, {N} entities total")

    if N == 0:
        return ConceptualModel(entities=[], relationships=[]), []

    # 2. Build Structural + Semantic representations
    fact_dict = {f.id: f.fact for f in all_facts}

    entity_texts = []
    for e in all_entities:
        e_facts = [fact_dict[fid] for fid in e.source_fact_ids if fid in fact_dict]
        attrs = [a.name for a in e.attributes]

        rep = f"{e.name}: {', '.join(attrs)} | {' '.join(e_facts)}"
        entity_texts.append(rep)

    embeddings = embed_texts(entity_texts)

    X = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if shard_map[i] == shard_map[j]:
                X[i, j] = -np.inf
                X[j, i] = -np.inf
            else:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                X[i, j] = sim
                X[j, i] = sim

    # Show similarity stats for cross-shard pairs
    cross_sims = X[(X > -100) & (X != 0)]
    if len(cross_sims) > 0:
        log(
            f"[Merger] {len(cross_sims)} cross-shard pairs, sim range [{cross_sims.min():.3f}, {cross_sims.max():.3f}], mean={cross_sims.mean():.3f}"
        )

    # 3. Merge-probability matrix via Beta Mixture
    P = compute_merge_probability_matrix(X, shard_map)

    # Boost exact name matches: same-name entities across shards are very likely the same concept
    for i in range(N):
        for j in range(i + 1, N):
            if (
                shard_map[i] != shard_map[j]
                and all_entities[i].name == all_entities[j].name
            ):
                boosted = max(P[i, j], 0.95)
                P[i, j] = boosted
                P[j, i] = boosted

    # 4. Correlation clustering on posterior probabilities
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = P[i, j]
            W[i, j] = p - 0.5
            W[j, i] = p - 0.5
    W[X == -np.inf] = -np.inf

    clusters = list(range(N))
    while True:
        best_merge = None
        best_gain = 0
        unique_c = list(set(clusters))
        for i, c1 in enumerate(unique_c):
            idx1 = [idx for idx, c in enumerate(clusters) if c == c1]
            for c2 in unique_c[i + 1 :]:
                idx2 = [idx for idx, c in enumerate(clusters) if c == c2]
                sub_W = W[np.ix_(idx1, idx2)]

                if np.any(sub_W == -np.inf):
                    gain = -np.inf
                else:
                    gain = float(np.sum(sub_W))

                if gain > best_gain:
                    best_gain = gain
                    best_merge = (c1, c2)
        if not best_merge:
            break
        c1, c2 = best_merge
        for i in range(N):
            if clusters[i] == c2:
                clusters[i] = c1

    # 5. Tension Flags -- pairs where posterior disagrees with cluster assignment
    for i in range(N):
        for j in range(i + 1, N):
            if X[i, j] == -np.inf:
                continue
            p = P[i, j]
            in_same = clusters[i] == clusters[j]
            if p > 0.5 and not in_same:
                flags.append(
                    ConflictFlag(
                        flag_type="VETOED_MERGE",
                        entities=[all_entities[i].name, all_entities[j].name],
                        posterior=float(p),
                        message=f"pair ({all_entities[i].name},{all_entities[j].name}) wanted to merge (P={p:.3f}) but was vetoed by cluster tension.",
                    )
                )
            elif p < 0.5 and in_same:
                flags.append(
                    ConflictFlag(
                        flag_type="FORCED_MERGE",
                        entities=[all_entities[i].name, all_entities[j].name],
                        posterior=float(p),
                        message=f"pair ({all_entities[i].name},{all_entities[j].name}) didn't want to merge (P={p:.3f}) but was dragged in.",
                    )
                )

    # 6. Build the Unified Schema
    unified_entities = []
    entity_remap = {}

    unique_clusters = list(set(clusters))
    for c_id in unique_clusters:
        members = [i for i, c in enumerate(clusters) if c == c_id]

        # Merge properties
        base = all_entities[members[0]]
        new_name = base.name  # Arbitrary base name

        all_attrs_dict = {}
        # Counter, not a set. The chosen identifier used to be
        # `list(all_ids)[0]` over a SET of tuples, i.e. whichever one Python's
        # hash order happened to put first -- so the merged entity's primary key
        # differed between runs on byte-identical input, since str hashing is
        # randomised per process. Counting lets the pick be both deterministic
        # and meaningful: the identifier the most shards agreed on.
        id_votes: Counter = Counter()
        all_facts_set = set()

        for m in members:
            e = all_entities[m]
            entity_remap[f"SHARD_{shard_map[m]}_{e.name}"] = new_name

            for attr in e.attributes:
                if attr.name not in all_attrs_dict:
                    all_attrs_dict[attr.name] = attr.model_copy(deep=True)
                else:
                    existing_attr = all_attrs_dict[attr.name]
                    existing_attr.source_fact_ids = list(
                        set(existing_attr.source_fact_ids) | set(attr.source_fact_ids)
                    )

            for f_id in e.source_fact_ids:
                all_facts_set.add(f_id)

            if e.identifier_attributes:
                id_votes[tuple(e.identifier_attributes)] += 1

        all_ids = set(id_votes)
        # Check ID conflict
        if len(all_ids) > 1:
            flags.append(
                ConflictFlag(
                    flag_type="IDENTIFIER_DISAGREEMENT",
                    entities=[new_name],
                    message=(
                        f"Entity {new_name} has conflicting IDs: "
                        f"{sorted(all_ids)}"
                    ),
                )
            )

        # Most-voted identifier wins; ties broken lexicographically so the
        # result depends only on the input, never on iteration order.
        chosen_id: Tuple[str, ...] = ()
        if id_votes:
            chosen_id = min(id_votes, key=lambda k: (-id_votes[k], k))
        merged_entity = Entity(
            name=new_name,
            attributes=list(all_attrs_dict.values()),
            identifier_attributes=list(chosen_id),
            source_fact_ids=sorted(all_facts_set),
        )
        unified_entities.append(merged_entity)

    # 7. Merge Relationships (Simple Overlay)
    all_rels = []
    for s_idx, shard in enumerate(shards):
        for r in shard.relationships:
            # Remap participants
            new_participants = []
            for p in r.participants:
                old_key = f"SHARD_{s_idx}_{p.entity}"
                new_ent_name = entity_remap.get(old_key, p.entity)
                new_participants.append(
                    Participant(
                        entity=new_ent_name,
                        role=p.role,
                        cardinality_min=p.cardinality_min,
                        cardinality_max=p.cardinality_max,
                    )
                )

            all_rels.append(
                Relationship(
                    name=r.name,
                    participants=new_participants,
                    degree=r.degree,
                    kind=r.kind,
                    attributes=r.attributes,
                    source_fact_ids=r.source_fact_ids,
                )
            )

    # Deduplicate relationships by name and exact participants
    unified_rels_dict = {}
    for r in all_rels:
        p_keys = tuple(sorted([p.entity for p in r.participants]))
        key = (r.name, p_keys)

        if key not in unified_rels_dict:
            unified_rels_dict[key] = r
        else:
            # Check cardinality contradictions
            existing = unified_rels_dict[key]
            if existing.kind != r.kind:
                flags.append(
                    ConflictFlag(
                        flag_type="CARDINALITY_CONTRADICTION",
                        entities=list(p_keys),
                        relationship=r.name,
                        message=f"Relationship {r.name} between {p_keys} has conflicting kinds ({existing.kind} vs {r.kind})",
                    )
                )

            unified_rels_dict[key].source_fact_ids.extend(r.source_fact_ids)
            unified_rels_dict[key].source_fact_ids = list(
                set(unified_rels_dict[key].source_fact_ids)
            )

    unified_rels = list(unified_rels_dict.values())

    # 8. Extract Semantic Flags (Attribute Synonyms & Cross-Category)
    # Uses the same Beta mixture approach: P(high-component | sim) > 0.5 → flag.
    # Near-0.5 posteriors → uncertain → LLM adjudicator decides.

    # Attribute Synonyms
    for e in unified_entities:
        attr_names = [a.name for a in e.attributes]
        if len(attr_names) > 1:
            attr_embs = embed_texts(attr_names)
            sims = []
            pairs = []
            for i in range(len(attr_names)):
                for j in range(i + 1, len(attr_names)):
                    sim = np.dot(attr_embs[i], attr_embs[j]) / (
                        np.linalg.norm(attr_embs[i]) * np.linalg.norm(attr_embs[j])
                    )
                    sims.append(float(sim))
                    pairs.append((attr_names[i], attr_names[j]))

            posteriors = compute_flag_posteriors(sims)
            for (a1, a2), p in zip(pairs, posteriors):
                if p > 0.5:
                    flags.append(
                        ConflictFlag(
                            flag_type="POSSIBLE_ATTR_SYNONYM",
                            entities=[e.name],
                            posterior=float(p),
                            message=f"'{a1}' and '{a2}' (posterior={p:.3f})",
                        )
                    )

    # Cross-Category
    if unified_entities and unified_rels:
        ent_names = [e.name for e in unified_entities]
        rel_names = [r.name for r in unified_rels]

        e_embs = embed_texts(ent_names)
        r_embs = embed_texts(rel_names)

        sims = []
        pairs = []
        for i, en in enumerate(ent_names):
            for j, rn in enumerate(rel_names):
                sim = np.dot(e_embs[i], r_embs[j]) / (
                    np.linalg.norm(e_embs[i]) * np.linalg.norm(r_embs[j])
                )
                sims.append(float(sim))
                pairs.append((en, rn))

        posteriors = compute_flag_posteriors(sims)
        for (en, rn), p in zip(pairs, posteriors):
            if p > 0.5:
                flags.append(
                    ConflictFlag(
                        flag_type="CROSS_CATEGORY_COLLISION",
                        entities=[en],
                        relationship=rn,
                        posterior=float(p),
                        message=f"Entity '{en}' vs Relationship '{rn}' (posterior={p:.3f})",
                    )
                )

    final_cm = ConceptualModel(entities=unified_entities, relationships=unified_rels)

    return final_cm, flags
