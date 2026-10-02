import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from safetensors.torch import load_model
import torch
import torch.nn as nn

from classification import train
from classification.classifier import build_classifier, get_last_logits
from model import GPT2, GPTConfig


def tiny_backbone():
    return GPT2(
        GPTConfig(
            vocab_size=32,
            d_model=16,
            n_heads=2,
            n_layers=2,
            n_embed=16,
            context_len=8,
            dropout=0.0,
        )
    )


def trainable_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def make_dataset(size, seed=0):
    """Random sequences whose label is carried by their last real token."""
    generator = torch.Generator().manual_seed(seed)
    examples = []
    for _ in range(size):
        length = int(torch.randint(1, 9, (1,), generator=generator))
        label = int(torch.randint(0, 4, (1,), generator=generator))
        ids = torch.randint(4, 32, (8,), generator=generator)
        ids[length - 1] = label
        examples.append((ids, length, torch.tensor(label)))
    return examples


class BuildClassifierTests(unittest.TestCase):
    def test_frozen_backbone_trains_only_the_head(self):
        model = build_classifier(tiny_backbone(), head="mlp", unfreeze="none")

        names = trainable_names(model)
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("lm_head.") for name in names))

    def test_last_block_unfreezes_the_final_block_and_layer_norm(self):
        model = build_classifier(tiny_backbone(), head="linear", unfreeze="last_block")

        prefixes = {name.split(".")[0] for name in trainable_names(model)}
        self.assertEqual(prefixes, {"lm_head", "transformers", "ln1"})
        blocks = {
            name.split(".")[1]
            for name in trainable_names(model)
            if name.startswith("transformers.")
        }
        # Only the last of the two blocks.
        self.assertEqual(blocks, {"1"})

    def test_all_unfreezes_every_parameter(self):
        model = build_classifier(tiny_backbone(), head="linear", unfreeze="all")

        self.assertEqual(
            trainable_names(model), {name for name, _ in model.named_parameters()}
        )

    def test_mlp_head_has_a_hidden_layer(self):
        model = build_classifier(tiny_backbone(), head="mlp")

        self.assertIsInstance(model.lm_head, nn.Sequential)
        self.assertEqual(model.lm_head[0].out_features, 64)
        self.assertEqual(model(torch.zeros(3, 8, dtype=torch.long)).shape, (3, 8, 4))

    def test_linear_head_is_a_single_layer(self):
        model = build_classifier(tiny_backbone(), head="linear", num_classes=7)

        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertEqual(model(torch.zeros(3, 8, dtype=torch.long)).shape, (3, 8, 7))

    def test_unknown_head_or_unfreeze_is_rejected(self):
        with self.assertRaises(ValueError):
            build_classifier(tiny_backbone(), head="conv")
        with self.assertRaises(ValueError):
            build_classifier(tiny_backbone(), unfreeze="first_block")


class GetLastLogitsTests(unittest.TestCase):
    def test_picks_the_last_real_token_of_each_sequence(self):
        logits = torch.arange(2 * 5 * 3, dtype=torch.float).view(2, 5, 3)

        picked = get_last_logits(logits, torch.tensor([2, 5]))

        self.assertEqual(picked.tolist(), [logits[0, 1].tolist(), logits[1, 4].tolist()])


class ResolveMaxStepsTests(unittest.TestCase):
    def test_derived_from_epochs_counts_the_short_final_batch(self):
        config = train.TrainingConfig(batch_size=32, n_epochs=3)

        self.assertEqual(train.resolve_max_steps(config, 100), 3 * 4)

    def test_explicit_max_steps_wins(self):
        config = train.TrainingConfig(batch_size=32, n_epochs=3, max_steps=7)

        self.assertEqual(train.resolve_max_steps(config, 100), 7)


class SaveConfigTests(unittest.TestCase):
    def test_config_round_trips_through_json(self):
        config = train.TrainingConfig(checkpoint_name="linear_head_full", unfreeze="all")
        with TemporaryDirectory() as directory:
            path = train.save_config(directory, config)
            saved = json.loads(path.read_text())

        self.assertEqual(path.name, "config.json")
        self.assertEqual(saved["checkpoint_name"], "linear_head_full")
        self.assertEqual(saved["unfreeze"], "all")
        self.assertEqual(train.TrainingConfig(**saved).max_lr, config.max_lr)


class EvaluateTests(unittest.TestCase):
    def test_short_final_batch_is_weighted_by_its_size(self):
        torch.manual_seed(0)
        model = build_classifier(tiny_backbone(), head="linear")
        dataset = make_dataset(5)

        # 5 examples in batches of 4 leaves a final batch of one.
        batched = train.evaluate(
            model, dataset, train.TrainingConfig(eval_batch_size=4), "cpu"
        )
        whole = train.evaluate(
            model, dataset, train.TrainingConfig(eval_batch_size=5), "cpu"
        )

        self.assertAlmostEqual(batched[0], whole[0], places=5)
        self.assertEqual(batched[1], whole[1])

    def test_leaves_the_model_in_training_mode(self):
        model = build_classifier(tiny_backbone(), head="linear")

        train.evaluate(model, make_dataset(3), train.TrainingConfig(), "cpu")

        self.assertTrue(model.training)


class TrainTests(unittest.TestCase):
    def run_training(self, directory, **overrides):
        torch.manual_seed(0)
        model = build_classifier(tiny_backbone(), head="linear", unfreeze="all")
        config = train.TrainingConfig(
            batch_size=16,
            eval_batch_size=16,
            max_lr=1e-2,
            min_lr=1e-3,
            warmup_steps=2,
            n_epochs=3,
            val_interval=4,
            log_interval=2,
            fused_optimizer=False,
            **overrides,
        )
        datasets = (make_dataset(64, seed=1), make_dataset(20, seed=2), make_dataset(20, seed=3))
        results = train.train(model, config, datasets, directory, "cpu")
        return model, config, datasets, results

    def test_derives_the_step_count_and_learns(self):
        with TemporaryDirectory() as directory:
            _, config, _, results = self.run_training(directory)

        self.assertEqual(config.max_steps, 12)
        # Four balanced classes start at ln(4) ~ 1.386.
        self.assertLess(results["best_val_loss"], 1.3)

    def test_test_numbers_come_from_the_saved_best_checkpoint(self):
        with TemporaryDirectory() as directory:
            _, config, datasets, results = self.run_training(directory)
            best = Path(directory) / "best.safetensors"
            self.assertTrue(best.exists())

            reloaded = build_classifier(tiny_backbone(), head="linear", unfreeze="all")
            load_model(reloaded, str(best))

        val_loss, _ = train.evaluate(reloaded, datasets[1], config, "cpu")
        test_loss, test_acc = train.evaluate(reloaded, datasets[2], config, "cpu")
        self.assertAlmostEqual(val_loss, results["best_val_loss"], places=5)
        self.assertAlmostEqual(test_loss, results["test_loss"], places=5)
        self.assertEqual(test_acc, results["test_acc"])

    def test_validates_on_the_final_step_even_off_the_interval(self):
        with TemporaryDirectory() as directory:
            _, _, _, results = self.run_training(directory, max_steps=3)

        # val_interval is 4, so step 3 is the only validation there is.
        self.assertEqual(results["best_step"], 3)


if __name__ == "__main__":
    unittest.main()
