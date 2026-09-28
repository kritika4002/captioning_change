"""
Dataset Statistics & Exploratory Data Analysis (EDA)

Generates comprehensive statistics and visualisations for the xBD dataset
to be used in the review presentation.

Outputs:
  - Console summary with total counts
  - Bar charts of damage distribution per event
  - Pie chart of overall damage breakdown
  - Event-wise sample count table
  - Saves all figures to outputs/eda/
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.caption_generator import CaptionGenerator

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping visualisations")


def count_dataset_stats(dataset_root: str) -> dict:
    """
    Walk the dataset and gather comprehensive statistics.

    Returns a dict with:
        - total_events, total_samples
        - per-event sample counts
        - damage distribution (global and per-event)
    """
    root = Path(dataset_root)
    stats = {
        "total_events": 0,
        "total_samples": 0,
        "events": {},
        "global_damage": Counter(),
        "per_event_damage": defaultdict(Counter),
        "disaster_types": Counter(),
    }

    gen = CaptionGenerator(seed=42)

    for event_dir in sorted(root.iterdir()):
        if not event_dir.is_dir() or event_dir.name == "xbd_subset":
            continue

        labels_dir = event_dir / "labels"
        images_dir = event_dir / "images"

        if not labels_dir.exists():
            continue

        stats["total_events"] += 1
        post_files = sorted(labels_dir.glob("*_post_disaster.json"))
        event_name = event_dir.name
        sample_count = 0

        for post_file in post_files:
            try:
                with open(post_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

            sample_count += 1
            metadata = data.get("metadata", {})
            disaster_type = metadata.get("disaster_type", "unknown")
            stats["disaster_types"][disaster_type] += 1

            # Count damage
            counts = gen._count_damage(data)
            for level, count in counts.items():
                stats["global_damage"][level] += count
                stats["per_event_damage"][event_name][level] += count

        stats["events"][event_name] = sample_count
        stats["total_samples"] += sample_count

    return stats


def print_summary(stats: dict):
    """Print a formatted summary to console."""
    print("\n" + "=" * 65)
    print("  xBD DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    print(f"\n  Total disaster events: {stats['total_events']}")
    print(f"  Total image pairs:     {stats['total_samples']}")
    total_buildings = sum(stats['global_damage'].values())
    print(f"  Total buildings:       {total_buildings:,}")

    print(f"\n  {'Event':<30} {'Samples':>8}")
    print("  " + "-" * 40)
    for event, count in sorted(stats['events'].items(), key=lambda x: -x[1]):
        print(f"  {event:<30} {count:>8}")

    print(f"\n  Global Damage Distribution:")
    print("  " + "-" * 40)
    for level in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
        count = stats['global_damage'].get(level, 0)
        pct = 100 * count / max(total_buildings, 1)
        bar = "#" * int(pct / 2)
        print(f"  {level:<15} {count:>8,} ({pct:5.1f}%) {bar}")

    print(f"\n  Disaster Types:")
    for dtype, count in stats['disaster_types'].most_common():
        print(f"    {dtype:<20} {count:>6}")

    print("=" * 65 + "\n")


def generate_plots(stats: dict, output_dir: str):
    """Generate and save EDA visualisation plots."""
    if not HAS_MPL:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Global Damage Pie Chart ──
    labels = ["No Damage", "Minor", "Major", "Destroyed"]
    sizes = [
        stats["global_damage"].get("no-damage", 0),
        stats["global_damage"].get("minor-damage", 0),
        stats["global_damage"].get("major-damage", 0),
        stats["global_damage"].get("destroyed", 0),
    ]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    explode = (0.02, 0.02, 0.04, 0.06)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=140,
        textprops={"fontsize": 12},
    )
    ax.set_title("Global Building Damage Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "damage_distribution_pie.png", dpi=150)
    plt.close()
    print(f"  Saved: {out / 'damage_distribution_pie.png'}")

    # ── 2. Per-Event Sample Count Bar Chart ──
    events = sorted(stats["events"].keys())
    counts = [stats["events"][e] for e in events]
    short_names = [e.replace("-", "\n") for e in events]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(events)), counts, color="#3498db", edgecolor="white")
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Image Pairs", fontsize=12)
    ax.set_title("Samples per Disaster Event", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out / "samples_per_event.png", dpi=150)
    plt.close()
    print(f"  Saved: {out / 'samples_per_event.png'}")

    # ── 3. Per-Event Stacked Damage Distribution ──
    damage_levels = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    level_colors = {"no-damage": "#2ecc71", "minor-damage": "#f1c40f",
                    "major-damage": "#e67e22", "destroyed": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(events))
    bottom = [0] * len(events)

    for level in damage_levels:
        vals = [stats["per_event_damage"][e].get(level, 0) for e in events]
        ax.bar(x, vals, bottom=bottom, label=level.replace("-", " ").title(),
               color=level_colors[level], edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Buildings", fontsize=12)
    ax.set_title("Damage Distribution by Event", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "damage_by_event_stacked.png", dpi=150)
    plt.close()
    print(f"  Saved: {out / 'damage_by_event_stacked.png'}")


def main():
    """Run EDA pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset EDA")
    parser.add_argument(
        "--data-root", type=str, default="./dataset",
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/eda",
        help="Directory to save EDA outputs",
    )
    args = parser.parse_args()

    print("Scanning dataset...")
    stats = count_dataset_stats(args.data_root)
    print_summary(stats)

    print("Generating plots...")
    generate_plots(stats, args.output_dir)

    # Save raw stats
    stats_path = Path(args.output_dir) / "dataset_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: (dict(v) if isinstance(v, (Counter, defaultdict)) else v)
        for k, v in stats.items()
    }
    serialisable["per_event_damage"] = {
        k: dict(v) for k, v in stats["per_event_damage"].items()
    }
    with open(stats_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved: {stats_path}")

    print("\n[OK] EDA complete!")


if __name__ == "__main__":
    main()
