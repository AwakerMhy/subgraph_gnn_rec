"""tests/test_curriculum.py — CurriculumScheduler 单元测试"""
import pytest
from src.recall.curriculum import CurriculumScheduler


class TestDifficulty:
    def test_warmup_returns_zero(self):
        sched = CurriculumScheduler(total_epochs=10, warmup_epochs=3)
        assert sched.difficulty(1) == pytest.approx(0.0)
        assert sched.difficulty(3) == pytest.approx(0.0)

    def test_linear_monotonic(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="linear")
        diffs = [sched.difficulty(e) for e in range(1, 11)]
        assert all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1))

    def test_cosine_monotonic(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="cosine")
        diffs = [sched.difficulty(e) for e in range(1, 11)]
        assert all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1))

    def test_step_monotonic(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="step")
        diffs = [sched.difficulty(e) for e in range(1, 11)]
        assert all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1))

    def test_first_epoch_near_zero(self):
        sched = CurriculumScheduler(total_epochs=100, schedule="linear")
        assert sched.difficulty(1) < 0.1

    def test_last_epoch_is_one(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="linear")
        assert sched.difficulty(10) == pytest.approx(1.0)

    def test_cosine_range(self):
        sched = CurriculumScheduler(total_epochs=20, schedule="cosine")
        for e in range(1, 21):
            d = sched.difficulty(e)
            assert 0.0 <= d <= 1.0

    def test_unknown_schedule_raises(self):
        with pytest.raises(AssertionError):
            CurriculumScheduler(total_epochs=10, schedule="unknown")


class TestTopKRange:
    def test_easy_range_at_tail(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="linear", warmup_epochs=10)
        # difficulty=0 → start should be near top_k//2
        start, end = sched.top_k_range(epoch=1, top_k=100)
        assert start >= 40  # in the tail half

    def test_hard_range_at_head(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="linear")
        # difficulty≈1 at last epoch → start=0
        start, end = sched.top_k_range(epoch=10, top_k=100)
        assert start == 0

    def test_range_within_bounds(self):
        sched = CurriculumScheduler(total_epochs=20, schedule="cosine")
        for e in range(1, 21):
            start, end = sched.top_k_range(e, top_k=100)
            assert 0 <= start < end <= 100

    def test_window_size_half_top_k(self):
        sched = CurriculumScheduler(total_epochs=10, schedule="linear")
        for e in range(1, 11):
            start, end = sched.top_k_range(e, top_k=100)
            assert end - start == 50  # window = top_k // 2
