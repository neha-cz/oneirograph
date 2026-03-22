node DreamSymbol {
    has label: str,
        symbol_type: str,
        features: list,
        emotional_valence: float,
        dream_date: str,
        recurrence_count: int = 0,
        transform_frequency: float = 0.0,
        confidence: float = 1.0;
}

node SkeletonNode {
    has dream_date: str,
        node_types: list,
        adjacency: list,
        embedding: list,
        is_anomaly: bool = False;
}

node DreamSession {
    has date: str,
        raw_text: str,
        summary: str = "";
}
