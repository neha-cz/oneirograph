include graph.schema;
include abilities.helpers;

walker NarrationWalker {
    has dream_date: str;

    def narrate(skeleton: SkeletonNode, recurring: list, anomaly: bool) -> str
        by llm(
            model="claude-sonnet-4-20250514",
            incl_info=(skeleton, recurring, anomaly)
        );

    can run with Root entry {
        anchor = root;
        skeletons = [(anchor) -->][?:SkeletonNode];
        today = None;
        for s in skeletons {
            if s.dream_date == self.dream_date {
                today = s;
                break;
            }
        }
        if not today {
            return;
        }

        recurring = find_recurring_patterns(today, skeletons);
        narration = self.narrate(today, recurring, today.is_anomaly);

        sessions = [(anchor) -->][?:DreamSession];
        for s in sessions {
            if s.date == self.dream_date {
                s.summary = narration;
            }
        }
    }
}
