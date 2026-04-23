"""src/online — 在线学习 + 社交网络生成模拟。"""

__all__ = [
    "OnlineEnv",
    "Feedback",
    "FeedbackSimulator",
    "StaticAdjacency",
    "run_online_simulation",
]


def __getattr__(name: str):
    if name == "OnlineEnv":
        from src.online.env import OnlineEnv
        return OnlineEnv
    if name in ("Feedback", "FeedbackSimulator"):
        import src.online.feedback as m
        return getattr(m, name)
    if name == "run_online_simulation":
        from src.online.loop import run_online_simulation
        return run_online_simulation
    if name == "StaticAdjacency":
        from src.online.static_adj import StaticAdjacency
        return StaticAdjacency
    raise AttributeError(f"module 'src.online' has no attribute {name!r}")
