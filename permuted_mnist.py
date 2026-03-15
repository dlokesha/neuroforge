"""
permuted_mnist.py — Permuted MNIST benchmark for continual learning.

TBC paper: "We evaluated our approach across 200 sequential tasks
using the Permuted MNIST benchmark."

What is Permuted MNIST?
  - Take MNIST (handwritten digits 0-9)
  - For each new task, apply a DIFFERENT fixed random pixel permutation
  - The labels stay the same but the pixel layout is scrambled differently
  - A network must learn task 2 without forgetting task 1, task 3 without
    forgetting tasks 1+2, and so on for 200 tasks
  - Standard backprop collapses — this is catastrophic forgetting in action
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class PermutedMNIST:
    """
    Generates a sequence of Permuted MNIST tasks.
    Each task is the same MNIST data with a different fixed pixel permutation.
    """

    def __init__(
        self,
        n_tasks: int = 200,
        n_train_per_task: int = 1000,
        n_test_per_task: int = 200,
        batch_size: int = 64,
        data_dir: str = "./data",
        seed: int = 42,
    ):
        self.n_tasks = n_tasks
        self.n_train_per_task = n_train_per_task
        self.n_test_per_task = n_test_per_task
        self.batch_size = batch_size
        self.seed = seed

        # Load base MNIST once
        print("Loading MNIST for continual learning benchmark...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        # Flatten to (N, 784)
        self.train_x = train_data.data.float().view(-1, 784) / 255.0
        self.train_y = train_data.targets
        self.test_x = test_data.data.float().view(-1, 784) / 255.0
        self.test_y = test_data.targets

        # Pre-generate all permutations — one per task
        rng = np.random.RandomState(seed)
        self.permutations = [
            torch.LongTensor(rng.permutation(784))
            for _ in range(n_tasks)
        ]
        # Task 0 is the identity permutation (original MNIST)
        self.permutations[0] = torch.arange(784)

        print(f"Ready: {n_tasks} tasks, {n_train_per_task} train / {n_test_per_task} test each")

    def get_task(self, task_id: int) -> tuple[DataLoader, DataLoader]:
        """
        Get train and test DataLoaders for a specific task.
        Each call applies that task's permutation to the pixel layout.
        """
        assert 0 <= task_id < self.n_tasks, f"Task {task_id} out of range"

        perm = self.permutations[task_id]

        # Apply permutation
        train_x_perm = self.train_x[:self.n_train_per_task, :][:, perm]
        test_x_perm = self.test_x[:self.n_test_per_task, :][:, perm]

        train_ds = TensorDataset(train_x_perm, self.train_y[:self.n_train_per_task])
        test_ds = TensorDataset(test_x_perm, self.test_y[:self.n_test_per_task])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        return train_loader, test_loader

    def get_all_test_loaders(self, up_to_task: int) -> list[DataLoader]:
        """
        Get test loaders for all tasks up to task_id.
        Used to measure retention — how well the network remembers past tasks.
        """
        return [self.get_task(t)[1] for t in range(up_to_task + 1)]