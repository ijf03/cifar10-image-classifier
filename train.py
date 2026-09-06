"""Train a small CPU CNN on 5,000 CIFAR-10 images; test on 1,000.

Run inside the Nix shell: python train.py --epochs 5
Torchvision downloads the full official archive (not Kaggle) to ./data once.
Only the selected, class-balanced subsets are used for training and testing.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def balanced_subset(dataset, images_per_class, seed):
    """Select reproducible random images from each of the ten classes."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor(dataset.targets)
    indices = []
    for label in range(10):
        candidates = torch.where(targets == label)[0]
        selected = candidates[torch.randperm(len(candidates), generator=generator)]
        indices.extend(selected[:images_per_class].tolist())
    return Subset(dataset, indices)


def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 10),  # Raw scores for CrossEntropyLoss; no softmax needed.
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("cifar10_cnn.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    torch.manual_seed(args.seed)
    # A small CPU model benefits from avoiding excessive thread overhead.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print("Downloading/checking the full CIFAR-10 archive in", args.data_dir, flush=True)
    train_data = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        balanced_subset(train_data, 500, args.seed), #500 images per category
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        balanced_subset(test_data, 100, args.seed), #100 images per category
        batch_size=args.batch_size, shuffle=False,
    )
    print("Using 5,000 training images and 1,000 test images on CPU.")
    print("Classes:", ", ".join(train_data.classes))

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (scores.argmax(dim=1) == labels).sum().item()
        count = len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / count:.4f}, train accuracy={correct / count:.1%}",
            flush=True,
        )

    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
    print(f"Test accuracy: {correct / len(test_loader.dataset):.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": train_data.classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "normalization_mean": (0.5, 0.5, 0.5),
        "normalization_std": (0.5, 0.5, 0.5),
    }, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
