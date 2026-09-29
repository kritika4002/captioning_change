import json
import random
from pathlib import Path
from collections import Counter

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
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def generate_caption(self, post_label_path, pre_label_path=None):
        with open(post_label_path, "r", encoding="utf-8") as f:
            post_data = json.load(f)

        meta = post_data.get("metadata", {})
        dis_name = meta.get("disaster", "unknown")
        dis_type = meta.get("disaster_type", "unknown")

        counts = Counter()
        feats = post_data.get("features", {})
        for k in ["lng_lat", "xy"]:
            buildings = feats.get(k, [])
            for b in buildings:
                sub = b.get("properties", {}).get("subtype", "un-classified")
                if sub in DAMAGE_LEVELS:
                    counts[sub] += 1
            if counts:
                break

        total = sum(counts.values())
        if total == 0:
            cap = self.rng.choice(NO_CHANGE_TEMPLATES)
            sev = 0.0
        else:
            weights = {"no-damage": 0.0, "minor-damage": 0.33, "major-damage": 0.67, "destroyed": 1.0}
            sev = round(sum(counts[k] * weights.get(k, 0) for k in counts) / total, 4)

            damaged = counts.get("minor-damage", 0) + counts.get("major-damage", 0) + counts.get("destroyed", 0)
            if damaged == 0:
                summary = f"all {total} buildings in the scene remain undamaged"
            else:
                pct = round(100 * damaged / total, 1)
                desc = (
                    f"widespread damage has affected {damaged} out of {total} buildings ({pct}%)" if pct >= 75
                    else f"significant damage is observed across {damaged} of {total} buildings ({pct}%)" if pct >= 40
                    else f"localised damage is detected in {damaged} of {total} buildings ({pct}%)" if pct >= 10
                    else f"minimal damage is observed with only {damaged} of {total} buildings affected ({pct}%)"
                )
                ctx = next((v for k, v in DISASTER_CONTEXT.items() if k in dis_type.lower()), None)
                summary = f"{desc}, {ctx}" if ctx else desc

            details = []
            for lvl in ["destroyed", "major-damage", "minor-damage", "no-damage"]:
                cnt = counts.get(lvl, 0)
                if cnt == 0:
                    continue
                d = self.rng.choice(DAMAGE_DESCRIPTIONS.get(lvl, ["is affected"]))
                if cnt == 1:
                    details.append(f"{cnt} building {d}")
                else:
                    plural_d = d.replace("has ", "have ").replace("is ", "are ").replace("shows", "show").replace("exhibits", "exhibit").replace("remains", "remain")
                    details.append(f"{cnt} buildings {plural_d}")

            if not details:
                detail_str = "No detailed structural information is available."
            elif len(details) == 1:
                detail_str = f"Specifically, {details[0]}."
            elif len(details) == 2:
                detail_str = f"Specifically, {details[0]}, while {details[1]}."
            else:
                detail_str = f"Specifically, {', '.join(details[:-1])}, and {details[-1]}."

            tmpl = self.rng.choice(SUMMARY_TEMPLATES)
            cap = tmpl.format(summary=summary, detail=detail_str)

        return {
            "caption": cap,
            "damage_counts": dict(counts),
            "disaster_type": dis_type,
            "disaster_name": dis_name,
            "total_buildings": total,
            "severity_score": sev,
        }

    def generate_captions_for_event(self, event_dir):
        ldir = Path(event_dir) / "labels"
        if not ldir.exists():
            raise FileNotFoundError(f"missing labels in {event_dir}")

        res = []
        for post_file in sorted(ldir.glob("*_post_disaster.json")):
            pre_file = post_file.parent / post_file.name.replace("_post_disaster", "_pre_disaster")
            pre_path = str(pre_file) if pre_file.exists() else None
            stem = post_file.stem.replace("_post_disaster", "")

            item = self.generate_caption(str(post_file), pre_path)
            item["sample_id"] = stem
            item["event"] = Path(event_dir).name
            res.append(item)
        return res
