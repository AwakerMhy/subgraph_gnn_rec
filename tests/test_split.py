"""tests/test_split.py — 时间切分函数单元测试"""
import numpy as np
import pandas as pd
import pytest

from src.utils.split import temporal_split, get_cutoff_times


def _make_edges(n: int = 100) -> pd.DataFrame:
    """生成有序的测试边列表。"""
    return pd.DataFrame({
        "src": np.random.randint(0, 20, size=n),
        "dst": np.random.randint(0, 20, size=n),
        "timestamp": np.sort(np.random.uniform(0, 100, size=n)),
    })


class TestTemporalSplit:
    def test_basic_split(self):
        edges = _make_edges(100)
        train, val, test = temporal_split(edges)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(edges)

    def test_no_future_leakage(self):
        """验证时间不泄露：val 的最小时间戳 >= train 的最大时间戳。"""
        edges = _make_edges(200)
        train, val, test = temporal_split(edges)
        assert val["timestamp"].min() >= train["timestamp"].max(), \
            "时间泄露：val 包含 train 时间范围内的边"
        assert test["timestamp"].min() >= val["timestamp"].max(), \
            "时间泄露：test 包含 val 时间范围内的边"

    def test_temporal_order_preserved(self):
        """验证切分后各子集内部时间戳有序。"""
        edges = _make_edges(150)
        train, val, test = temporal_split(edges)
        for split in [train, val, test]:
            ts = split["timestamp"].values
            assert (ts[:-1] <= ts[1:]).all(), "切分后时间戳不单调"

    def test_ratio_approximately_correct(self):
        edges = _make_edges(1000)
        train, val, test = temporal_split(edges, train_ratio=0.7, val_ratio=0.15)
        assert 0.65 <= len(train) / len(edges) <= 0.75
        assert 0.10 <= len(val) / len(edges) <= 0.20
        assert 0.10 <= len(test) / len(edges) <= 0.20

    def test_small_dataset(self):
        """3条边的极小数据集不应抛异常。"""
        edges = pd.DataFrame({
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "timestamp": [1.0, 2.0, 3.0],
        })
        train, val, test = temporal_split(edges)
        assert len(train) + len(val) + len(test) == 3

    def test_missing_timestamp_column_raises(self):
        bad_df = pd.DataFrame({"src": [0], "dst": [1]})
        with pytest.raises(AssertionError, match="timestamp"):
            temporal_split(bad_df)

    def test_get_cutoff_times(self):
        edges = _make_edges(100)
        train, val, test = temporal_split(edges)
        cutoffs = get_cutoff_times(train, val, test)
        assert "train_cutoff" in cutoffs
        assert "val_cutoff" in cutoffs
        assert "test_cutoff" in cutoffs
        assert cutoffs["train_cutoff"] <= cutoffs["val_cutoff"] <= cutoffs["test_cutoff"]


class TestSelfLoopFiltering:
    """测试 base.py 中的自环过滤。"""

    def test_remove_self_loops(self):
        from src.dataset.base import TemporalDataset

        edges = pd.DataFrame({
            "src": [0, 1, 2, 2],
            "dst": [1, 2, 2, 3],  # 第三行是自环
            "timestamp": [1.0, 2.0, 3.0, 4.0],
        })
        result = TemporalDataset._remove_self_loops(edges)
        assert len(result) == 3
        assert (result["src"] != result["dst"]).all()


class TestNodeIdRemap:
    def test_remap_to_contiguous(self):
        from src.dataset.base import TemporalDataset

        edges = pd.DataFrame({
            "src": [10, 20, 30],
            "dst": [20, 30, 10],
            "timestamp": [1.0, 2.0, 3.0],
        })
        remapped, mapping = TemporalDataset._remap_node_ids(edges)
        all_ids = set(remapped["src"]) | set(remapped["dst"])
        assert all_ids == {0, 1, 2}
        assert max(all_ids) == len(mapping) - 1


class TestNormalizeTimestamps:
    def test_range_is_zero_to_one(self):
        from src.dataset.base import TemporalDataset

        edges = pd.DataFrame({
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "timestamp": [100.0, 200.0, 300.0],
        })
        result = TemporalDataset._normalize_timestamps(edges)
        assert result["timestamp"].min() == pytest.approx(0.0)
        assert result["timestamp"].max() == pytest.approx(1.0)
        assert "timestamp_raw" in result.columns
        assert result["timestamp_raw"].tolist() == [100.0, 200.0, 300.0]


class TestSyntheticGenerators:
    def test_sbm_generate(self):
        from src.dataset.synthetic.sbm import SBMGenerator

        gen = SBMGenerator(n_nodes=50, n_communities=5, T=20, edges_per_step=3, seed=0)
        df = gen.generate()
        assert set(df.columns) == {"src", "dst", "timestamp"}
        assert len(df) > 0
        assert (df["src"] != df["dst"]).all(), "生成了自环"
        assert (df["timestamp"].diff().dropna() >= 0).all(), "时间戳不单调"

        feats = gen.get_node_features()
        assert feats.shape == (50, 5 + 8)
        assert feats.dtype == "float32"

    def test_hawkes_generate(self):
        from src.dataset.synthetic.hawkes import HawkesGenerator

        gen = HawkesGenerator(n_nodes=30, T=10.0, seed=1)
        df = gen.generate()
        assert len(df) > 0
        feats = gen.get_node_features()
        assert feats.shape[0] == 30

    def test_triadic_generate(self):
        from src.dataset.synthetic.triadic import TriadicGenerator

        gen = TriadicGenerator(n_nodes=50, T=30, seed=2)
        df = gen.generate()
        assert len(df) > 0
        feats = gen.get_node_features()
        assert feats.shape[0] == 50
