include graph.schema;

import from backends.tebd_backend { policy_select_action, policy_tebd_update }

walker AdaptiveWalker {
    has trajectory: list = [];
    has reward: float = 0.0;

    can meta with Root entry {
        anchor = root;
        skels = [(anchor) -->][?:SkeletonNode];
        if not skels {
            return;
        }
        for h in skels {
            self.trajectory.append(
                {
                    "node_types": h.node_types,
                    "embedding": h.embedding,
                    "is_anomaly": h.is_anomaly
                }
            );
            cur = h;
            nbrs = [(cur) -->][?:SkeletonNode];
            embs = [];
            for n in nbrs {
                embs.append(n.embedding);
            }
            if embs {
                _ = policy_select_action(h.embedding, embs);
            }
            connectivity = len(nbrs);
            anomaly_penalty = 0.0;
            if h.is_anomaly {
                anomaly_penalty = -2.0;
            }
            self.reward += float(connectivity) * 0.1 + anomaly_penalty;
        }
        policy_tebd_update(self.trajectory, self.reward);
    }
}
