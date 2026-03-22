include graph.schema;
include abilities.helpers;

import from backends.qnn_backend { classify_node }

walker DreamIngester {
    has dream_data: dict;
    has dream_date: str;

    can ingest with Root entry {
        session = DreamSession(date=self.dream_date, raw_text=self.dream_data["raw_text"], summary="");
        root ++> session;

        anchor = root;
        existing_nodes = [(anchor) -->][?:DreamSymbol];
        id_map = {};
        nodes_payload = self.dream_data["nodes"];

        for node_data in nodes_payload {
            feat = node_data["features"];
            cls = classify_node(feat);
            sym_type = cls["type"];
            conf = cls["confidence"];

            merged = False;
            for existing in existing_nodes {
                if existing.symbol_type == sym_type {
                    sim = cosine_similarity(existing.features, feat);
                    if sim > 0.85 {
                        existing.recurrence_count += 1;
                        session ++> existing;
                        id_map[node_data["id"]] = existing;
                        merged = True;
                        break;
                    }
                }
            }

            if not merged {
                new_node = DreamSymbol(
                    label=node_data["label"],
                    symbol_type=sym_type,
                    features=feat,
                    emotional_valence=node_data["emotional_valence"],
                    dream_date=self.dream_date,
                    recurrence_count=0,
                    transform_frequency=0.0,
                    confidence=conf
                );
                root ++> new_node;
                session ++> new_node;
                existing_nodes.append(new_node);
                id_map[node_data["id"]] = new_node;
            }
        }

        for edge_data in self.dream_data["edges"] {
            from_id = edge_data["from"];
            to_id = edge_data["to"];
            if from_id not in id_map or to_id not in id_map {
                continue;
            }
            from_node = id_map[from_id];
            to_node = id_map[to_id];
            conf = 1.0;
            if "confidence" in edge_data {
                conf = edge_data["confidence"];
            }
            from_node +>: DreamEdge(relationship=edge_data["relationship"], confidence=conf) :+> to_node;
        }
    }
}
