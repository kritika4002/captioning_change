"""
Caption Generator — Converts xBD JSON damage annotations into
natural language change captions.

This module is a key part of the data pipeline: it reads the building
damage labels (from the xBD dataset JSON files) and synthesizes
descriptive natural language captions that a VLM would learn to generate.

Each caption describes:
  - Overall damage distribution
  - Specific structural changes
  - Disaster type context
  - Spatial extent of damage
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ── Damage level hierarchy (ordinal) ──
DAMAGE_LEVELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": -1,
}

DAMAGE_DESCRIPTIONS = {
    "no-damage": [
        "appears intact with no visible structural damage",
        "shows no signs of damage or deterioration",
        "remains structurally sound with no observable changes",
        "is undamaged and maintains its original condition",
    ],
    "minor-damage": [
        "has sustained minor structural damage with slight surface-level changes",
        "shows signs of partial damage including possible roof displacement",
        "exhibits minor deterioration with limited structural compromise",
        "has experienced light damage with small visible changes to the exterior",
    ],
    "major-damage": [
        "has suffered significant structural damage with visible collapse",
        "shows major deterioration with substantial portions compromised",
        "exhibits severe damage with large sections of the structure affected",
        "has sustained heavy damage resulting in partial structural failure",
    ],
    "destroyed": [
        "has been completely destroyed with no recognisable structure remaining",
        "is fully demolished leaving only debris and rubble",
        "has been entirely razed with total structural collapse",
        "shows complete devastation with the building reduced to ruins",
    ],
}

DISASTER_CONTEXT = {
    "hurricane": "as a result of hurricane-force winds and storm surge",
    "flooding": "due to severe flooding and water inundation",
    "tornado": "caused by tornado wind damage and debris impact",
    "earthquake": "resulting from seismic activity and ground shaking",
    "tsunami": "due to tsunami wave impact and subsequent flooding",
    "volcano": "caused by volcanic eruption, lava flow, or ashfall",
    "wildfire": "as a result of wildfire and intense heat exposure",
    "fire": "as a result of wildfire and intense heat exposure",
    "bushfire": "as a result of bushfire and intense heat exposure",
}

# ── Templates for generating varied captions ──
SUMMARY_TEMPLATES = [
    "In this bi-temporal satellite image pair, {summary}. {detail}",
    "Comparing the pre- and post-disaster imagery reveals that {summary}. {detail}",
    "The post-disaster image shows that {summary}. {detail}",
    "Analysis of the before-and-after satellite images indicates that {summary}. {detail}",
    "Upon examination of the temporal image pair, it is observed that {summary}. {detail}",
]

NO_CHANGE_TEMPLATES = [
    "No significant structural changes are detected between the pre- and post-disaster images. The area appears to have been unaffected by the disaster event.",
    "The satellite imagery shows no discernible damage or structural changes in this area. All visible buildings remain intact.",
    "Comparing the two temporal images reveals no observable changes. The structures in this region appear to have withstood the disaster without damage.",
]


class CaptionGenerator:
    """
    Generates natural language change captions from xBD JSON annotations.

    Each xBD label JSON contains building polygons with damage subtypes.
    This class aggregates that information and synthesises descriptive
    captions that serve as ground-truth for training the captioning model.
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible caption variation.
        """
        self.rng = random.Random(seed)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def generate_caption(
        self,
        post_label_path: str,
        pre_label_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate a caption from a post-disaster label JSON.

        Args:
            post_label_path: Path to the post-disaster JSON label file.
            pre_label_path: Optional path to the pre-disaster JSON label.

        Returns:
            dict with keys:
                - caption (str): Generated natural language caption
                - damage_counts (dict): Count per damage level
                - disaster_type (str): Type of disaster
                - total_buildings (int): Total annotated buildings
                - severity_score (float): Normalised severity 0-1
        """
        post_data = self._load_label(post_label_path)
        pre_data = self._load_label(pre_label_path) if pre_label_path else None

        metadata = post_data.get("metadata", {})
        disaster_name = metadata.get("disaster", "unknown")
        disaster_type = metadata.get("disaster_type", "unknown")

        # Extract building damage from the post-disaster annotations
        damage_counts = self._count_damage(post_data)
        total_buildings = sum(damage_counts.values())

        if total_buildings == 0:
            caption = self.rng.choice(NO_CHANGE_TEMPLATES)
            severity_score = 0.0
        else:
            severity_score = self._compute_severity(damage_counts, total_buildings)
            summary = self._build_summary(damage_counts, total_buildings, disaster_type)
            detail = self._build_detail(damage_counts, disaster_type, disaster_name)
            template = self.rng.choice(SUMMARY_TEMPLATES)
            caption = template.format(summary=summary, detail=detail)

        return {
            "caption": caption,
            "damage_counts": dict(damage_counts),
            "disaster_type": disaster_type,
            "disaster_name": disaster_name,
            "total_buildings": total_buildings,
            "severity_score": severity_score,
        }

    def generate_captions_for_event(
        self, event_dir: str
    ) -> List[Dict]:
        """
        Generate captions for all image pairs in a disaster event directory.

        Args:
            event_dir: Path to an event directory
                       (e.g., dataset/hurricane-harvey)

        Returns:
            List of caption dicts, one per image pair.
        """
        event_path = Path(event_dir)
        labels_dir = event_path / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Collect unique sample IDs from post-disaster label files
        post_files = sorted(labels_dir.glob("*_post_disaster.json"))
        results = []

        for post_file in post_files:
            # Derive the matching pre-disaster label path
            pre_file = post_file.parent / post_file.name.replace(
                "_post_disaster", "_pre_disaster"
            )
            pre_path = str(pre_file) if pre_file.exists() else None

            # Derive the image file stem for pairing
            stem = post_file.stem.replace("_post_disaster", "")

            caption_data = self.generate_caption(str(post_file), pre_path)
            caption_data["sample_id"] = stem
            caption_data["event"] = event_path.name
            results.append(caption_data)

        return results

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _load_label(self, path: str) -> Dict:
        """Load and parse a JSON label file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _count_damage(self, label_data: Dict) -> Counter:
        """Count buildings per damage level from a label JSON."""
        counts = Counter()
        features = label_data.get("features", {})

        # xBD stores features under lng_lat and xy keys
        for coord_key in ["lng_lat", "xy"]:
            buildings = features.get(coord_key, [])
            for building in buildings:
                subtype = building.get("properties", {}).get("subtype", "un-classified")
                if subtype in DAMAGE_LEVELS:
                    counts[subtype] += 1

            # Only count from one coordinate system to avoid double-counting
            if counts:
                break

        return counts

    def _compute_severity(self, counts: Counter, total: int) -> float:
        """
        Compute a normalised severity score ∈ [0, 1].

        Weighted average: no-damage=0, minor=0.33, major=0.67, destroyed=1.0
        """
        if total == 0:
            return 0.0

        weights = {
            "no-damage": 0.0,
            "minor-damage": 0.33,
            "major-damage": 0.67,
            "destroyed": 1.0,
            "un-classified": 0.0,
        }
        weighted_sum = sum(counts[k] * weights.get(k, 0) for k in counts)
        return round(weighted_sum / total, 4)

    def _build_summary(
        self, counts: Counter, total: int, disaster_type: str
    ) -> str:
        """Build the high-level summary portion of the caption."""
        parts = []

        # Determine dominant damage category
        damaged = (
            counts.get("minor-damage", 0)
            + counts.get("major-damage", 0)
            + counts.get("destroyed", 0)
        )
        undamaged = counts.get("no-damage", 0)

        if damaged == 0:
            return f"all {total} buildings in the scene remain undamaged"

        damage_pct = round(100 * damaged / total, 1)

        if damage_pct >= 75:
            parts.append(
                f"widespread damage has affected {damaged} out of {total} "
                f"buildings ({damage_pct}%)"
            )
        elif damage_pct >= 40:
            parts.append(
                f"significant damage is observed across {damaged} of {total} "
                f"buildings ({damage_pct}%)"
            )
        elif damage_pct >= 10:
            parts.append(
                f"localised damage is detected in {damaged} of {total} "
                f"buildings ({damage_pct}%)"
            )
        else:
            parts.append(
                f"minimal damage is observed with only {damaged} of {total} "
                f"buildings affected ({damage_pct}%)"
            )

        # Add disaster context
        for key, context in DISASTER_CONTEXT.items():
            if key in disaster_type.lower():
                parts.append(context)
                break

        return ", ".join(parts)

    def _build_detail(
        self, counts: Counter, disaster_type: str, disaster_name: str
    ) -> str:
        """Build the detailed breakdown portion of the caption."""
        detail_parts = []

        for level in ["destroyed", "major-damage", "minor-damage", "no-damage"]:
            count = counts.get(level, 0)
            if count == 0:
                continue

            desc = self.rng.choice(DAMAGE_DESCRIPTIONS.get(level, ["is affected"]))

            if count == 1:
                detail_parts.append(f"{count} building {desc}")
            else:
                detail_parts.append(f"{count} buildings {desc.replace('has ', 'have ').replace('is ', 'are ').replace('shows', 'show').replace('exhibits', 'exhibit').replace('remains', 'remain')}")

        if not detail_parts:
            return "No detailed structural information is available."

        # Join with proper punctuation
        if len(detail_parts) == 1:
            return f"Specifically, {detail_parts[0]}."
        elif len(detail_parts) == 2:
            return f"Specifically, {detail_parts[0]}, while {detail_parts[1]}."
        else:
            joined = ", ".join(detail_parts[:-1])
            return f"Specifically, {joined}, and {detail_parts[-1]}."


# ── CLI entry point for testing ──
if __name__ == "__main__":
    import sys

    gen = CaptionGenerator(seed=42)

    if len(sys.argv) > 1:
        result = gen.generate_caption(sys.argv[1])
        print(f"\n{'='*60}")
        print(f"Caption: {result['caption']}")
        print(f"Damage: {result['damage_counts']}")
        print(f"Severity: {result['severity_score']}")
        print(f"{'='*60}")
    else:
        print("Usage: python caption_generator.py <post_disaster_label.json>")
