include graph.schema;
include abilities.helpers;

import from backends.perceval_backend { run_qubo }

walker QuboWalker {
    has dream_date: str;

    can extract with Root entry {
        anchor = root;
        sessions = [(anchor) -->][?:DreamSession];
        today_session = None;
        for s in sessions {
            if s.date == self.dream_date {
                today_session = s;
                break;
            }
        }
        if not today_session {
            return;
        }

        ts = today_session;
        session_nodes = [(ts) -->][?:DreamSymbol];
        if not session_nodes {
            return;
        }

        n_len = len(session_nodes);
        node_features = [session_nodes[i].features for i in range(n_len)];
        node_types = [session_nodes[i].symbol_type for i in range(n_len)];
        edges = [];
        for i in range(n_len) {
            n = session_nodes[i];
            cur = n;
            for e in [(cur) -->][?:DreamEdge] {
                for j in range(n_len) {
                    if e.target == session_nodes[j] {
                        edges.append([i, j]);
                    }
                }
            }
        }

        skel_idx = run_qubo(node_features, edges);
        skeleton_nodes = [];
        for k in skel_idx {
            skeleton_nodes.append(session_nodes[int(k)]);
        }
        skeleton_types = [n.symbol_type for n in skeleton_nodes];

        skeleton_edges = [];
        for pair in edges {
            i = int(pair[0]);
            j = int(pair[1]);
            if i in skel_idx and j in skel_idx {
                skeleton_edges.append(pair);
            }
        }

        embedding = compute_skeleton_embedding(skeleton_types, skeleton_edges);

        skeleton = SkeletonNode(
            dream_date=self.dream_date,
            node_types=skeleton_types,
            adjacency=skeleton_edges,
            embedding=embedding,
            is_anomaly=False
        );
        root ++> skeleton;

        prior_skeletons = [(anchor) -->][?:SkeletonNode];
        others = [];
        for p in prior_skeletons {
            if p.dream_date != self.dream_date {
                others.append(p);
            }
        }

        for prior in others {
            skeleton +>: SkeletonEdge(edge_type="temporal", similarity_score=0.0) :+> prior;
            sim = cosine_similarity(embedding, prior.embedding);
            if sim > 0.7 {
                skeleton +>: SkeletonEdge(edge_type="structurally_similar", similarity_score=sim) :+> prior;
            }
            shared = 0;
            for t in skeleton_types {
                if t in prior.node_types {
                    shared += 1;
                }
            }
            if shared >= 2 {
                skeleton +>: SkeletonEdge(edge_type="shares_node_type", similarity_score=float(shared)) :+> prior;
            }
        }

        if len(others) > 5 {
            sims = [];
            for p in others {
                sims.append(cosine_similarity(embedding, p.embedding));
            }
            avg_sim = mean(sims);
            if avg_sim < 0.3 {
                skeleton.is_anomaly = True;
            }
        }
    }
}
