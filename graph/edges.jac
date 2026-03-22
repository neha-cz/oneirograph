edge DreamEdge {
    has relationship: str,
        confidence: float = 1.0;
}

edge SkeletonEdge {
    has edge_type: str,
        similarity_score: float = 0.0;
}
