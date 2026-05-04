from pathlib import Path
import csv
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
CACHE_PATH = BASE / "reference_lira_loss_cache.npz"

OUTPUT_CSV = BASE / "submission.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 9
BATCH_SIZE = 128


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
    def __init__(self, pub_ds, priv_ds):
        self.pub_ds = pub_ds
        self.priv_ds = priv_ds
        self.n_pub = len(pub_ds)

    def __len__(self):
        return len(self.pub_ds) + len(self.priv_ds)

    def __getitem__(self, index):
        if index < self.n_pub:
            id_, img, label, membership = self.pub_ds[index]
            return index, img, label

        priv_index = index - self.n_pub
        id_, img, label, membership = self.priv_ds[priv_index]
        return index, img, label


def build_target_model():
    model = resnet18(weights=None)

    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model.to(DEVICE)

    return model


# Measure how wrong the model is
def negative_loss_score(logits, labels):
    losses = F.cross_entropy(logits, labels, reduction="none")
    return -losses

# compute target model's negative loss score for each sample in the combined dataset (public + private)
def compute_target_loss_scores(target_model, combined_ds):
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

# Compute Gaussian log pdf values; to compare whether score is IN-like or OUT-like 
def gaussian_logpdf(x, mean, std):
    std = max(float(std), 1e-6)
    return -0.5 * np.log(2.0 * np.pi * std * std) - ((x - mean) ** 2) / (2.0 * std * std)

# Compute classwise Gaussian LiRA scores
def classwise_pooled_lira_scores(
    target_scores,
    ref_scores_matrix,
    inclusion_matrix,
    labels,
):
    final_scores = np.zeros(len(target_scores), dtype=np.float64)

    labels = np.asarray(labels, dtype=np.int64)

    print("\nClasswise Gaussian LiRA reference distribution stats")
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

# This adjusts the LiRA score by subtracting the sample’s own OUT difficulty so easy samples don't get high scores.
def rmia_normalized_lira_scores(
    target_scores,
    ref_scores_matrix,
    inclusion_matrix,
    labels,
):
    """
    Starts with Gaussian LiRA:
        log P(target_score | IN, class) - log P(target_score | OUT, class)

    Then subtracts a per-sample OUT difficulty baseline estimated from
    reference models where that same sample was OUT.
    """

    final_scores = np.zeros(len(target_scores), dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    print("\nComputing RMIA-normalized LiRA scores...")

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

        x = target_scores[class_mask]

        log_in = gaussian_logpdf(x, in_mean, in_std)
        log_out = gaussian_logpdf(x, out_mean, out_std)

        lira = log_in - log_out

        n_samples = class_ref_scores.shape[0]
        population_baseline = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            out_mask = ~class_inclusion[i]

            if out_mask.sum() > 0:
                pop_scores = class_ref_scores[i, out_mask]
                pop_log_out = gaussian_logpdf(pop_scores, out_mean, out_std)
                population_baseline[i] = pop_log_out.mean()
            else:
                population_baseline[i] = 0.0

        final_scores[class_mask] = lira - population_baseline

    return final_scores


def normalize_to_0_1(scores):
    scores = np.asarray(scores, dtype=np.float64)

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def rank_normalize(scores):
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)

    if len(scores) == 1:
        return np.array([1.0])

    return ranks / (len(scores) - 1)


# How many same-class samples does x beat by a margin?
def population_rmia_scores(base_log_ratios, labels, gamma):
    """
    True RMIA-style population comparison.

    For each sample x and same-class population samples z:
        score(x) = fraction of z where log_ratio_x > log_ratio_z + log(gamma)

    Higher = more member-like.
    """

    labels = np.asarray(labels, dtype=np.int64)
    base_log_ratios = np.asarray(base_log_ratios, dtype=np.float64)

    log_gamma = np.log(gamma)
    scores = np.zeros(len(base_log_ratios), dtype=np.float64)

   #  If x beats many same-class samples, x is more suspicious/member-like.
    for c in range(NUM_CLASSES):
        class_mask = labels == c
        class_values = np.sort(base_log_ratios[class_mask])

        x = base_log_ratios[class_mask]
        thresholds = x - log_gamma

        counts = np.searchsorted(class_values, thresholds, side="left")
        scores[class_mask] = counts / len(class_values)

    return scores

def save_submission(ids, scores, output_csv):
    scores = normalize_to_0_1(scores)

    assert len(ids) == len(scores), "IDs and scores length mismatch"
    assert len(set(ids.tolist())) == len(ids), "Duplicate IDs found"
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores are not in [0,1]"

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score"])

        for id_, score in zip(ids, scores):
            writer.writerow([int(id_), float(score)])

    print("\nSaved:", output_csv)
    print("Rows:", len(ids))
    print("Unique IDs:", len(set(ids.tolist())))
    print("Score min:", scores.min())
    print("Score max:", scores.max())

    print("\nFirst 5 rows:")
    for id_, score in list(zip(ids, scores))[:5]:
        print(int(id_), float(score))


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

print("Public samples:", n_pub)
print("Private samples:", n_priv)
print("Combined samples:", len(combined_ds))

pub_labels = np.array(pub_ds.labels, dtype=np.int64)
priv_labels = np.array(priv_ds.labels, dtype=np.int64)
priv_ids = np.array(priv_ds.ids)

combined_labels = np.concatenate([pub_labels, priv_labels], axis=0)

print("\nLoading reference LiRA cache...")
data = np.load(CACHE_PATH)

ref_scores_matrix = data["ref_scores_matrix"]
inclusion_matrix = data["inclusion_matrix"]
completed_models = data["completed_models"]

print("Reference score matrix:", ref_scores_matrix.shape)
print("Inclusion matrix:", inclusion_matrix.shape)
print("Completed reference models:", int(completed_models.sum()))

assert completed_models.all(), "Not all reference models are completed. Finish diagnostic first."

print("\nLoading target model...")
target_model = build_target_model()
print("Target model loaded.")

print("\nComputing target -loss scores...")
target_loss_scores = compute_target_loss_scores(
    target_model=target_model,
    combined_ds=combined_ds,
)

print("\nComputing Gaussian LiRA scores...")
gaussian_scores = classwise_pooled_lira_scores(
    target_scores=target_loss_scores,
    ref_scores_matrix=ref_scores_matrix,
    inclusion_matrix=inclusion_matrix,
    labels=combined_labels,
)

print("\nComputing RMIA-normalized LiRA scores...")
rmia_scores = rmia_normalized_lira_scores(
    target_scores=target_loss_scores,
    ref_scores_matrix=ref_scores_matrix,
    inclusion_matrix=inclusion_matrix,
    labels=combined_labels,
)

print("\nCombining scores: current90_pop10_gamma_1.05")

gaussian_rank = rank_normalize(gaussian_scores)
rmia_rank = rank_normalize(rmia_scores)

current_best_scores = 0.90 * gaussian_rank + 0.10 * rmia_rank

population_scores = population_rmia_scores(
    base_log_ratios=gaussian_scores,
    labels=combined_labels,
    gamma=1.05,
)

population_rank = rank_normalize(population_scores)

combined_scores = 0.875 * rank_normalize(current_best_scores) + 0.125 * population_rank

private_scores = combined_scores[n_pub:]

save_submission(
    ids=priv_ids,
    scores=private_scores,
    output_csv=OUTPUT_CSV,
)