import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import OCRDataset
from src.models.crnn import CRNN


def build_dataset(data_dir: Path, max_label_length: int = 20):
    image_paths = []
    labels = []

    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        for file_path in sorted(category_dir.iterdir()):
            if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            label = file_path.stem
            if len(label) <= max_label_length:
                image_paths.append(str(file_path))
                labels.append(label)

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    chars = sorted(set("".join(labels)))
    char2idx = {c: i for i, c in enumerate(chars)}
    blank_idx = len(chars)
    num_classes = len(chars) + 1

    dataset = OCRDataset(image_paths, labels, char2idx)
    return dataset, chars, char2idx, blank_idx, num_classes


def collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    batch_images = torch.stack(batch_images)
    label_lengths = torch.tensor([len(label) for label in batch_labels], dtype=torch.long)
    batch_labels = torch.cat(batch_labels).long()
    return batch_images, batch_labels, label_lengths


def move_batch_to_device(batch, device, non_blocking=False):
    batch_images, batch_labels, batch_label_lengths = batch
    batch_images = batch_images.to(device, non_blocking=non_blocking)
    batch_labels = batch_labels.to(device, non_blocking=non_blocking)
    batch_label_lengths = batch_label_lengths.to(device, non_blocking=non_blocking)
    return batch_images, batch_labels, batch_label_lengths


def compute_ctc_batch_loss(model, batch, criterion, device, non_blocking=False):
    batch_images, batch_labels, batch_label_lengths = move_batch_to_device(
        batch, device, non_blocking=non_blocking
    )
    outputs = model(batch_images)
    log_probs = outputs.permute(1, 0, 2).log_softmax(2)
    input_lengths = torch.full(
        size=(batch_images.size(0),),
        fill_value=log_probs.size(0),
        dtype=torch.long,
        device=device,
    )
    loss = criterion(log_probs, batch_labels, input_lengths, batch_label_lengths)
    return loss


def run_epoch(model, data_loader, optimizer, criterion, device, train=True, non_blocking=False):
    model.train(mode=train)
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in data_loader:
            loss = compute_ctc_batch_loss(
                model, batch, criterion, device, non_blocking=non_blocking
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)


def save_checkpoint(path: Path, model, optimizer, epoch, train_loss, val_loss, chars, blank_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "chars": chars,
            "blank_idx": blank_idx,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN baseline for historical document OCR")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/dataset"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-label-length", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"

    dataset, chars, char2idx, blank_idx, num_classes = build_dataset(
        args.data_dir, max_label_length=args.max_label_length
    )

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    model = CRNN(num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(chars)} | Blank index: {blank_idx}")

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True, non_blocking=use_pin_memory
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False, non_blocking=use_pin_memory
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        last_ckpt = args.checkpoint_dir / "last_crnn.pt"
        save_checkpoint(last_ckpt, model, optimizer, epoch + 1, train_loss, val_loss, chars, blank_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_ckpt = args.checkpoint_dir / "best_crnn.pt"
            save_checkpoint(best_ckpt, model, optimizer, epoch + 1, train_loss, val_loss, chars, blank_idx)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | lr={current_lr:.6f}"
        )

    model.load_state_dict(best_state_dict)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Train loss history: {[round(x, 4) for x in history['train_loss']]}")
    print(f"Val loss history  : {[round(x, 4) for x in history['val_loss']]}")


if __name__ == "__main__":
    main()
