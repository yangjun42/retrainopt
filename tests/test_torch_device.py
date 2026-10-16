"""CUDA device routing without hardware, plus optional real RNG checks."""

from contextlib import contextmanager, ExitStack
from importlib.util import find_spec
import unittest
from unittest.mock import Mock, patch

from retrainopt.adapters import TorchModelAdapter


@unittest.skipUnless(find_spec("torch") is not None, "torch is optional")
class TorchDeviceTests(unittest.TestCase):
    def test_cuda_rng_guard_and_training_use_the_same_resolved_device(self):
        import torch

        for requested, expected in (("cuda", 2), ("cuda:0", 0), ("cuda:1", 1)):
            for fail in (False, True):
                with self.subTest(device=requested, fail=fail):
                    current = [2]
                    rng_states = {0: 100, 1: 101, 2: 102}
                    original_states = dict(rng_states)
                    cpu_state = torch.random.get_rng_state().clone()
                    seeded = []
                    protected = []
                    restored = []

                    @contextmanager
                    def device_context(index):
                        previous = current[0]
                        current[0] = index
                        try:
                            yield
                        finally:
                            current[0] = previous

                    def seed(value):
                        seeded.append((current[0], value))
                        rng_states[current[0]] = value

                    def get_rng(index):
                        protected.append(index)
                        return rng_states[index]

                    def set_rng(state, index):
                        restored.append(index)
                        rng_states[index] = state

                    model = Mock()
                    model.to.return_value = model
                    inputs, targets = Mock(), Mock()
                    inputs.to.return_value, targets.to.return_value = inputs, targets

                    def forward(_inputs):
                        device = model.to.call_args.args[0]
                        index = device.index if isinstance(device, torch.device) else torch.device(device).index
                        index = current[0] if index is None else index
                        rng_states[index] += 1
                        torch.rand(1)
                        if fail:
                            raise RuntimeError("forward failed")
                        return Mock()

                    model.side_effect = forward
                    optimizer = Mock()
                    adapter = TorchModelAdapter(lambda: model, optimizer_factory=lambda params: optimizer, epochs=1, device=requested)
                    with ExitStack() as stack:
                        stack.enter_context(patch("torch.cuda.current_device", side_effect=lambda: current[0]))
                        stack.enter_context(patch("torch.cuda.device", side_effect=device_context))
                        stack.enter_context(patch("torch.cuda.manual_seed", side_effect=seed))
                        stack.enter_context(patch("torch.cuda.get_rng_state", side_effect=get_rng))
                        stack.enter_context(patch("torch.cuda.set_rng_state", side_effect=set_rng))
                        synchronize = stack.enter_context(patch("torch.cuda.synchronize"))
                        stack.enter_context(patch("torch.utils.data.DataLoader", return_value=[(inputs, targets)]))
                        stack.enter_context(patch("torch.nn.CrossEntropyLoss", return_value=Mock()))
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                                adapter.fit([], step=1, n_samples=2, random_seed=7)
                        else:
                            self.assertIs(adapter.fit([], step=1, n_samples=2, random_seed=7), model)
                    resolved = torch.device("cuda", expected)
                    model.to.assert_called_once_with(resolved)
                    inputs.to.assert_called_once_with(resolved)
                    targets.to.assert_called_once_with(resolved)
                    self.assertEqual(seeded, [(expected, 7)])
                    self.assertEqual(protected, [expected])
                    self.assertEqual(restored, [expected])
                    self.assertEqual(rng_states, original_states)
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_state))
                    self.assertEqual(current[0], 2)
                    if fail:
                        synchronize.assert_not_called()
                    else:
                        synchronize.assert_called_once_with(resolved)

    def test_evaluation_resolves_unindexed_cuda_for_both_tensors(self):
        import torch

        inputs, targets = Mock(), Mock()
        inputs.to.return_value, targets.to.return_value = inputs, targets
        inputs.size.return_value = 2
        model = Mock()
        model.modules.return_value = []
        loss = Mock()
        loss.item.return_value = 1.25
        with patch("torch.cuda.current_device", return_value=2) as current, \
                patch("torch.utils.data.DataLoader", return_value=[(inputs, targets)]), \
                patch("torch.nn.CrossEntropyLoss", return_value=Mock(return_value=loss)):
            value = TorchModelAdapter(lambda: model, device="cuda").evaluate(model, [], step=1)
        self.assertEqual(value, 1.25)
        current.assert_called_once_with()
        inputs.to.assert_called_once_with(torch.device("cuda", 2))
        targets.to.assert_called_once_with(torch.device("cuda", 2))

    def test_real_cuda_dropout_is_reproducible_and_restores_all_rng_states(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA hardware is unavailable; device routing is covered by mocks")
        devices = list(range(torch.cuda.device_count()))
        selected = 1 if len(devices) > 1 else 0
        data = torch.utils.data.TensorDataset(
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]]), torch.tensor([0, 1, 0, 1]),
        )
        with torch.random.fork_rng(devices=devices), torch.cuda.device(selected):
            cpu_state = torch.random.get_rng_state().clone()
            cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
            factory = lambda: torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 2))
            adapter = TorchModelAdapter(
                factory, optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01), epochs=2, batch_size=2, device="cuda",
            )
            first = adapter.fit(data, step=1, n_samples=4, random_seed=7)
            second = adapter.fit(data, step=1, n_samples=4, random_seed=7)
            for left, right in zip(first.parameters(), second.parameters()):
                self.assertEqual(left.device, torch.device("cuda", selected))
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            adapter.evaluate(first, data, step=1)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_state))
            for before, after in zip(cuda_states, torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
