from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pretraining import train


def make_checkpoint(root, run_name, step):
    """Create an empty state checkpoint file inside a run directory."""
    path = Path(root) / run_name / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class PathTests(unittest.TestCase):
    def test_run_directory_is_named_and_timestamped(self):
        config = train.TrainingConfig()
        directory = train.run_directory(
            "checkpoints", config, now=datetime(2026, 9, 30, 14, 30, 22)
        )

        self.assertEqual(directory, Path("checkpoints/gpt2_20260930_143022"))

    def test_run_directory_uses_the_configured_name(self):
        config = train.TrainingConfig(checkpoint_name="gpt2-medium")
        directory = train.run_directory(
            "checkpoints", config, now=datetime(2026, 9, 30, 14, 30, 22)
        )

        self.assertEqual(directory.name, "gpt2-medium_20260930_143022")

    def test_state_checkpoints_are_zero_padded(self):
        path = train.state_checkpoint_path("run", 42)

        self.assertEqual(path, Path("run/checkpoints/step_000042.pt"))
        # Zero padding is what makes lexicographic order match step order.
        self.assertLess(
            train.state_checkpoint_path("run", 9).name,
            train.state_checkpoint_path("run", 10).name,
        )

    def test_model_checkpoint_carries_the_step(self):
        config = train.TrainingConfig()
        path = train.model_checkpoint_path("run", config, 5000)

        self.assertEqual(path, Path("run/gpt2_step_005000.safetensors"))


class FindLatestCheckpointTests(unittest.TestCase):
    def test_returns_none_when_nothing_is_saved(self):
        config = train.TrainingConfig()
        with TemporaryDirectory() as directory:
            self.assertIsNone(train.find_latest_checkpoint(directory, config))

    def test_picks_the_highest_step_of_the_newest_run(self):
        config = train.TrainingConfig()
        with TemporaryDirectory() as directory:
            make_checkpoint(directory, "gpt2_20260930_090000", 1000)
            make_checkpoint(directory, "gpt2_20260930_090000", 2000)
            make_checkpoint(directory, "gpt2_20260930_181205", 500)
            latest = make_checkpoint(directory, "gpt2_20260930_181205", 3000)

            found = train.find_latest_checkpoint(directory, config)

        # Newer run wins even though the older one reached step 2000 first.
        self.assertEqual(found, latest)

    def test_ignores_runs_with_a_different_checkpoint_name(self):
        with TemporaryDirectory() as directory:
            make_checkpoint(directory, "gpt2-medium_20261001_120000", 9000)
            mine = make_checkpoint(directory, "gpt2_20260930_090000", 1000)

            found = train.find_latest_checkpoint(
                directory, train.TrainingConfig(checkpoint_name="gpt2")
            )

        self.assertEqual(found, mine)


class OffsetSamplerTests(unittest.TestCase):
    def test_zero_offset_is_plain_sequential_order(self):
        self.assertEqual(list(train.OffsetSampler(5)), [0, 1, 2, 3, 4])

    def test_offset_starts_there_and_wraps(self):
        self.assertEqual(list(train.OffsetSampler(5, offset=2)), [2, 3, 4, 0, 1])

    def test_covers_every_index_exactly_once(self):
        sampler = train.OffsetSampler(97, offset=40)
        indices = list(sampler)

        self.assertEqual(len(indices), 97)
        self.assertEqual(sorted(indices), list(range(97)))
        self.assertEqual(len(sampler), 97)

    def test_offset_larger_than_the_dataset_wraps_around(self):
        self.assertEqual(list(train.OffsetSampler(4, offset=10)), [2, 3, 0, 1])

    def test_empty_dataset_yields_nothing(self):
        self.assertEqual(list(train.OffsetSampler(0, offset=3)), [])


class PruneCheckpointsTests(unittest.TestCase):
    def test_keeps_only_the_newest_n(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "gpt2_20260930_090000"
            for step in (1000, 2000, 3000, 4000, 5000):
                make_checkpoint(directory, run.name, step)

            removed = train.prune_checkpoints(run, 3)

            remaining = sorted(p.name for p in (run / "checkpoints").iterdir())

        self.assertEqual(len(removed), 2)
        self.assertEqual(
            remaining,
            ["step_003000.pt", "step_004000.pt", "step_005000.pt"],
        )

    def test_keeps_everything_when_under_the_limit(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "gpt2_20260930_090000"
            for step in (1000, 2000):
                make_checkpoint(directory, run.name, step)

            removed = train.prune_checkpoints(run, 3)

            self.assertEqual(removed, [])
            self.assertEqual(len(list((run / "checkpoints").iterdir())), 2)

    def test_non_positive_keeps_everything(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "gpt2_20260930_090000"
            for step in (1000, 2000, 3000):
                make_checkpoint(directory, run.name, step)

            self.assertEqual(train.prune_checkpoints(run, 0), [])
            self.assertEqual(len(list((run / "checkpoints").iterdir())), 3)

    def test_leaves_the_final_weights_alone(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "gpt2_20260930_090000"
            for step in (1000, 2000, 3000):
                make_checkpoint(directory, run.name, step)
            weights = run / "gpt2_step_003000.safetensors"
            weights.touch()

            train.prune_checkpoints(run, 1)

            # Pruning only ever removes step_*.pt under checkpoints/.
            self.assertTrue(weights.exists())

    def test_never_touches_another_run(self):
        with TemporaryDirectory() as directory:
            mine = Path(directory) / "gpt2_20260930_090000"
            for step in (1000, 2000, 3000):
                make_checkpoint(directory, mine.name, step)
            other = make_checkpoint(directory, "gpt2_20261001_120000", 1000)

            train.prune_checkpoints(mine, 1)

            self.assertTrue(other.exists())
            self.assertEqual(len(list((mine / "checkpoints").iterdir())), 1)

    def test_survives_a_run_with_no_checkpoints_yet(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "gpt2_20260930_090000"
            (run / "checkpoints").mkdir(parents=True)

            self.assertEqual(train.prune_checkpoints(run, 3), [])


class UsableExamplesTests(unittest.TestCase):
    def test_rounds_down_to_whole_batches(self):
        # 7 examples at batch 4 yields one batch; the last 3 are dropped.
        self.assertEqual(train.usable_examples(7, 4), 4)

    def test_exact_multiple_keeps_everything(self):
        self.assertEqual(train.usable_examples(8, 4), 8)

    def test_fewer_examples_than_a_batch_yields_none(self):
        self.assertEqual(train.usable_examples(3, 4), 0)


class ResumeConfigTests(unittest.TestCase):
    def test_accepts_an_identical_config(self):
        config = train.TrainingConfig()
        # Should not raise.
        train.check_resume_config(train.asdict(config), config)

    def test_rejects_a_changed_token_budget(self):
        config = train.TrainingConfig()
        saved = train.asdict(config) | {"total_batch_size": 1024}

        with self.assertRaisesRegex(ValueError, "total_batch_size"):
            train.check_resume_config(saved, config)

    def test_rejects_a_changed_context_length(self):
        config = train.TrainingConfig()
        saved = train.asdict(config) | {"context_len": 512}

        with self.assertRaisesRegex(ValueError, "context_len"):
            train.check_resume_config(saved, config)

    def test_allows_a_changed_max_steps(self):
        config = train.TrainingConfig(max_steps=9_000)
        saved = train.asdict(config) | {"max_steps": 5_000}

        # Extending a run is normal, so this warns rather than raising.
        train.check_resume_config(saved, config)


if __name__ == "__main__":
    unittest.main()
