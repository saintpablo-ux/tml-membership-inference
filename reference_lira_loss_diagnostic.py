from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18


BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"

CACHE_PATH = BASE / "reference_lira_loss_cache.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 9
BATCH_SIZE = 128

# Serious reference-LiRA diagnostic settings.
NUM_REF_MODELS = 32
REF_EPOCHS = 10
REF_LR = 1e-3
TRAIN_FRACTION = 0.5

BASE_SEED = 2026


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


class CombinedReferenceDataset(Dataset):
    """
    Combined pub + priv dataset.

    global_idx:
        0 ... len(pub)-1 are public samples
        len(pub) ... len(pub)+len(priv)-1 are private samples

    Reference models train on this combined population.
    """

    def __init__(self, pub_ds, priv_ds):
        self.pub_ds = pub_ds
        self.priv_ds = priv_ds
        self.n_pub = len(pub_ds)
        self.n_priv = len(priv_ds)

    def __len__(self):
        return self.n_pub + self.n_priv

    def __getitem__(self, index):
        if index < self.n_pub:
            id_, img, label, membership = self.pub_ds[index]
            return index, img, label
        else:
            priv_index = index - self.n_pub
            id_, img, label, membership = self.priv_ds[priv_index]
            return index, img, label


def build_resnet18():
    model = resnet18(weights=None)

    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, NUM_CLASSES)

    return model


def build_target_model():
    model = build_resnet18()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model.to(DEVICE)
    return model


def negative_loss_score(logits, labels):
    """
    Main signal:
        lower cross-entropy loss = more member-like
        so score = -loss
    """

    losses = F.cross_entropy(logits, labels, reduction="none")
    return -losses


def train_reference_model(combined_ds, train_indices, model_id):
    seed = BASE_SEED + model_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_resnet18().to(DEVICE)
    model.train()

    train_subset = Subset(combined_ds, train_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=REF_LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining reference model {model_id + 1}/{NUM_REF_MODELS}")
    print("Reference train samples:", len(train_indices))

    for epoch in range(1, REF_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for global_idx, imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        print(
            f"Ref model {model_id + 1} epoch {epoch:02d}/{REF_EPOCHS} | "
            f"loss: {avg_loss:.4f} | train acc: {acc:.4f}"
        )

    model.eval()
    return model


def evaluate_reference_model(model, combined_ds):
    """
    Evaluate one reference model on all pub+priv samples.

    Returns:
        scores[global_idx] = -cross_entropy_loss
    """

    loader = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    scores = np.zeros(len(combined_ds), dtype=np.float64)

    model.eval()

    with torch.no_grad():
        for global_idx, imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            batch_scores = negative_loss_score(logits, labels)

            scores[global_idx.numpy()] = batch_scores.cpu().numpy()

    return scores


def compute_target_loss_scores(target_model, combined_ds):
    """
    Compute target model -loss score on all pub+priv samples.
    """

    loader = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    scores = np.zeros(len(combined_ds), dtype=np.float64)

    target_model.eval()

    with torch.no_grad():
        for global_idx, imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = target_model(imgs)
            batch_scores = negative_loss_score(logits, labels)

            scores[global_idx.numpy()] = batch_scores.cpu().numpy()

    return scores


def gaussian_logpdf(x, mean, std):
    std = max(float(std), 1e-6)
    return -0.5 * np.log(2.0 * np.pi * std * std) - ((x - mean) ** 2) / (2.0 * std * std)


def classwise_pooled_lira_scores(
    target_scores,
    ref_scores_matrix,
    inclusion_matrix,
    labels,
):
    """
    Classwise pooled reference LiRA.

    For each class:
        collect all reference scores where sample was IN reference training
        collect all reference scores where sample was OUT reference training

    Then for each target sample:
        score = log P(target_score | IN, class)
              - log P(target_score | OUT, class)

    Higher = more member-like.
    """

    n = len(target_scores)
    final_scores = np.zeros(n, dtype=np.float64)

    labels = np.asarray(labels, dtype=np.int64)

    print("\nClasswise reference distribution stats")
    print("class | IN count | OUT count | IN mean | OUT mean | IN std | OUT std")

    for c in range(NUM_CLASSES):
        class_mask = labels == c

        class_ref_scores = ref_scores_matrix[class_mask]
        class_inclusion = inclusion_matrix[class_mask]

        in_values = class_ref_scores[class_inclusion]
        out_values = class_ref_scores[~class_inclusion]

        in_mean = in_values.mean()
        in_std = in_values.std() + 1e-6

        out_mean = out_values.mean()
        out_std = out_values.std() + 1e-6

        print(
            f"{c:5d} | "
            f"{len(in_values):8d} | "
            f"{len(out_values):9d} | "
            f"{in_mean:.6f} | "
            f"{out_mean:.6f} | "
            f"{in_std:.6f} | "
            f"{out_std:.6f}"
        )

        x = target_scores[class_mask]

        log_in = gaussian_logpdf(x, in_mean, in_std)
        log_out = gaussian_logpdf(x, out_mean, out_std)

        final_scores[class_mask] = log_in - log_out

    return final_scores


def tpr_at_fpr(y_true, scores, target_fpr=0.05):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    total_members = np.sum(y_true == 1)
    total_nonmembers = np.sum(y_true == 0)

    max_false_positives = int(np.floor(target_fpr * total_nonmembers))

    tp = 0
    fp = 0
    best_tpr = 0.0

    for membership in y_sorted:
        if membership == 1:
            tp += 1
        else:
            fp += 1

        if fp <= max_false_positives:
            best_tpr = max(best_tpr, tp / total_members)
        else:
            break

    return best_tpr


def topk_precision(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    sorted_y = y_true[order]
    k = min(k, len(sorted_y))
    return sorted_y[:k].sum() / k


def evaluate_public(name, y_true, scores):
    tpr = tpr_at_fpr(y_true, scores, target_fpr=0.05)
    top100 = topk_precision(y_true, scores, k=100)
    top350 = topk_precision(y_true, scores, k=350)

    print(
        f"{name:30s} "
        f"TPR@5%FPR={tpr:.6f} "
        f"top100={top100:.4f} "
        f"top350={top350:.4f}"
    )

    return tpr, top100, top350


def make_inclusion_matrix(n_total):
    """
    Precompute which samples are IN each reference model.
    Deterministic, so cache/resume stays consistent.
    """

    rng = np.random.default_rng(BASE_SEED)
    all_indices = np.arange(n_total)
    train_size = int(TRAIN_FRACTION * n_total)

    inclusion = np.zeros((n_total, NUM_REF_MODELS), dtype=bool)

    for model_id in range(NUM_REF_MODELS):
        train_indices = rng.choice(
            all_indices,
            size=train_size,
            replace=False,
        )
        inclusion[train_indices, model_id] = True

    return inclusion


def save_cache(ref_scores_matrix, completed_models, inclusion_matrix):
    np.savez_compressed(
        CACHE_PATH,
        ref_scores_matrix=ref_scores_matrix,
        completed_models=completed_models,
        inclusion_matrix=inclusion_matrix,
    )
    print("Saved cache:", CACHE_PATH)


def load_or_initialize_cache(n_total, inclusion_matrix):
    if CACHE_PATH.exists():
        print("Loading existing cache:", CACHE_PATH)
        data = np.load(CACHE_PATH)

        ref_scores_matrix = data["ref_scores_matrix"]
        completed_models = data["completed_models"]
        cached_inclusion = data["inclusion_matrix"]

        if ref_scores_matrix.shape != (n_total, NUM_REF_MODELS):
            raise ValueError("Cache shape mismatch. Delete old cache and rerun.")

        if not np.array_equal(cached_inclusion, inclusion_matrix):
            raise ValueError("Inclusion matrix mismatch. Delete old cache and rerun.")

        return ref_scores_matrix, completed_models

    print("No cache found. Starting fresh.")

    ref_scores_matrix = np.zeros((n_total, NUM_REF_MODELS), dtype=np.float64)
    completed_models = np.zeros(NUM_REF_MODELS, dtype=bool)

    return ref_scores_matrix, completed_models


print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading datasets...")

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

pub_ds.transform = transform
priv_ds.transform = transform

n_pub = len(pub_ds)
n_priv = len(priv_ds)

combined_ds = CombinedReferenceDataset(pub_ds, priv_ds)
n_total = len(combined_ds)

print("Public samples:", n_pub)
print("Private samples:", n_priv)
print("Combined samples:", n_total)

pub_memberships = np.array(pub_ds.membership, dtype=np.int64)
pub_labels = np.array(pub_ds.labels, dtype=np.int64)
priv_labels = np.array(priv_ds.labels, dtype=np.int64)

combined_labels = np.concatenate([pub_labels, priv_labels], axis=0)

print("\nPreparing inclusion matrix...")
inclusion_matrix = make_inclusion_matrix(n_total)

ref_scores_matrix, completed_models = load_or_initialize_cache(
    n_total=n_total,
    inclusion_matrix=inclusion_matrix,
)

for model_id in range(NUM_REF_MODELS):
    if completed_models[model_id]:
        print(f"\nReference model {model_id + 1}/{NUM_REF_MODELS} already completed. Skipping.")
        continue

    train_indices = np.where(inclusion_matrix[:, model_id])[0]

    ref_model = train_reference_model(
        combined_ds=combined_ds,
        train_indices=train_indices,
        model_id=model_id,
    )

    print(f"Evaluating reference model {model_id + 1}/{NUM_REF_MODELS} on all samples...")
    ref_scores = evaluate_reference_model(ref_model, combined_ds)

    ref_scores_matrix[:, model_id] = ref_scores
    completed_models[model_id] = True

    save_cache(
        ref_scores_matrix=ref_scores_matrix,
        completed_models=completed_models,
        inclusion_matrix=inclusion_matrix,
    )

    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll reference models completed.")

print("\nLoading target model...")
target_model = build_target_model()
print("Target model loaded.")

print("\nComputing target -loss scores...")
target_loss_scores = compute_target_loss_scores(
    target_model=target_model,
    combined_ds=combined_ds,
)

print("\nComputing classwise pooled reference LiRA scores...")
lira_scores = classwise_pooled_lira_scores(
    target_scores=target_loss_scores,
    ref_scores_matrix=ref_scores_matrix,
    inclusion_matrix=inclusion_matrix,
    labels=combined_labels,
)

pub_loss_scores = target_loss_scores[:n_pub]
pub_lira_scores = lira_scores[:n_pub]

print("\nPublic diagnostic")
loss_result = evaluate_public("-loss baseline", pub_memberships, pub_loss_scores)
lira_result = evaluate_public("reference LiRA -loss", pub_memberships, pub_lira_scores)

print("\nDecision rule")
print("Continue only if reference LiRA beats -loss on TPR@5%FPR, top100, and top350.")

if (
    lira_result[0] > loss_result[0]
    and lira_result[1] > loss_result[1]
    and lira_result[2] > loss_result[2]
):
    print("Reference LiRA branch is promising.")
else:
    print("Do not submit yet: reference LiRA did not beat -loss on all public criteria.")