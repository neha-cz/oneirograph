"""Python helpers for Jac endpoints."""


def filter_anomaly_skeletons(nodes):  # Jac graph list; avoid typing for interop
    return [n for n in nodes if getattr(n, "is_anomaly", False)]
