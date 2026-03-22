include graph.schema;

import from abilities.helpers_py {
    cosine_similarity as _cosine_similarity,
    compute_skeleton_embedding as _compute_skeleton_embedding,
    mean as _mean,
    top_by_similarity as _top_by_similarity
}

def cosine_similarity(a: list, b: list) -> float {
    return _cosine_similarity(a, b);
}

def compute_skeleton_embedding(node_types: list, edges: list) -> list {
    return _compute_skeleton_embedding(node_types, edges);
}

def find_recurring_patterns(target: SkeletonNode, all_skeletons: list) -> list {
    recurring = [];
    for s in all_skeletons {
        if s.dream_date == target.dream_date {
            continue;
        }
        sim = cosine_similarity(target.embedding, s.embedding);
        if sim > 0.75 {
            recurring.append({"date": s.dream_date, "similarity": sim});
        }
    }
    return _top_by_similarity(recurring, 5);
}

def find_in(nodes: list, node_id: str) -> any {
    for n in nodes {
        if str(n.jid) == node_id {
            return n;
        }
        if n.label == node_id {
            return n;
        }
    }
    return None;
}

def mean(values: list) -> float {
    return _mean(values);
}
