"""Small PyTorch classification example. Install retrainopt[torch] first."""

import torch
from torch.utils.data import TensorDataset

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import TorchModelAdapter, TorchTimelineAdapter


def main() -> None:
    torch.set_num_threads(1)
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    blocks = [TensorDataset(features, labels), TensorDataset(features, 1 - labels)]
    timeline = TorchTimelineAdapter(blocks, blocks)
    adapter = TorchModelAdapter(
        lambda: torch.nn.Linear(1, 2), epochs=2, batch_size=4, device="cpu",
    )
    config = PlannerConfig(n_choices=(1.0,), initial_loss=1.0, time_cost_weight=0.0)
    result = RetrainingPlanner(timeline, adapter, config).plan("exact")
    print([decision.action.value for decision in result.decisions])
    print(f"Selected schedule loss: {result.loss_cost:.6f}")


if __name__ == "__main__":
    main()
