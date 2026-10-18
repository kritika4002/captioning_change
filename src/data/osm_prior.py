"""
OpenStreetMap (OSM) Footprint Prior & Spatial Disaster Ledger
============================================================

Operational Disaster Assessment Engine:
In real-world disasters (e.g., FEMA, UN OCHA, Copernicus EMS), post-disaster
ground-truth labels DO NOT exist in real time. However, pre-disaster building
vector footprints are globally available for free via OpenStreetMap (OSM).

This module:
1. Ingests pre-disaster building vector footprint priors (OSM GeoJSON or xBD pre-labels).
2. Partitions the scene into 4 tactical emergency response quadrants (NW, NE, SW, SE).
3. Samples the vision-language model's spatial change attribution heatmap inside each footprint.
4. Produces an automated, label-free building damage classification ledger and triage priority.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .damage_overlay import _parse_wkt_polygon, DAMAGE_COLORS


SECTOR_NAMES = ["NW", "NE", "SW", "SE"]
SECTOR_FULL_NAMES = {
    "NW": "Northwest Sector",
    "NE": "Northeast Sector",
    "SW": "Southwest Sector",
    "SE": "Southeast Sector",
}


def get_sector_for_point(x, y, width=1024, height=1024):
    """Assigns (x, y) coordinates to one of 4 emergency response quadrants."""
    mid_x = width / 2.0
    mid_y = height / 2.0
    if y < mid_y:
        return "NW" if x < mid_x else "NE"
    else:
        return "SW" if x < mid_x else "SE"


class OSMBuildingPrior:
    """
    Manages pre-disaster building footprint priors.
    """
    def __init__(self, footprint_path=None, image_size=(1024, 1024)):
        self.width, self.height = image_size
        self.buildings = []
        if footprint_path:
            self.load_footprints(footprint_path)

    def load_footprints(self, footprint_path):
        """Loads building polygons from xBD pre-disaster label or standard OSM GeoJSON."""
        path = Path(footprint_path)
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        self.buildings = []
        features = data.get("features", {})
        xy_features = features.get("xy", [])

        # Fallback to standard GeoJSON features list if "xy" is not a key
        if not xy_features and isinstance(features, list):
            xy_features = features
        elif not xy_features and "features" in data and isinstance(data["features"], list):
            xy_features = data["features"]

        for idx, feat in enumerate(xy_features):
            pts = []
            if "wkt" in feat:
                pts = _parse_wkt_polygon(feat["wkt"])
            elif "geometry" in feat:
                coords = feat["geometry"].get("coordinates", [])
                if coords and len(coords[0]) >= 3:
                    pts = [(float(c[0]), float(c[1])) for c in coords[0] if len(c) >= 2]

            if len(pts) < 3:
                continue

            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            sector = get_sector_for_point(cx, cy, self.width, self.height)

            props = feat.get("properties", {})
            uid = props.get("uid", f"bld_{idx:04d}")

            self.buildings.append({
                "id": uid,
                "pts": pts,
                "centroid": (cx, cy),
                "sector": sector,
            })

        return self.buildings

    def get_sector_distribution(self):
        """Returns structure count per tactical quadrant."""
        counts = {s: 0 for s in SECTOR_NAMES}
        for b in self.buildings:
            counts[b["sector"]] += 1
        return counts


class AutomatedDamageLedger:
    """
    Fuses pre-disaster footprint priors with the model's visual attention/change
    heatmap to generate a structured operational disaster damage ledger.
    """
    def __init__(self, osm_prior):
        self.osm_prior = osm_prior

    def assess_damage_from_saliency(self, saliency_map):
        """
        Samples the change saliency heatmap at each building footprint.

        Saliency map is a 2D numpy array [H, W] normalized to [0, 1].
        Damage thresholds:
          < 0.25 -> No Damage
          0.25 - 0.50 -> Minor Damage
          0.50 - 0.75 -> Major Damage
          >= 0.75 -> Destroyed
        """
        h_map, w_map = saliency_map.shape
        sx = w_map / float(self.osm_prior.width)
        sy = h_map / float(self.osm_prior.height)

        sector_stats = {
            s: {"intact": 0, "minor": 0, "major": 0, "destroyed": 0, "total": 0}
            for s in SECTOR_NAMES
        }

        assessed_buildings = []

        for b in self.osm_prior.buildings:
            pts = b["pts"]
            # Sample saliency at vertices + centroid
            sampled_values = []
            cx, cy = b["centroid"]
            mx = int(np.clip(cx * sx, 0, w_map - 1))
            my = int(np.clip(cy * sy, 0, h_map - 1))
            sampled_values.append(float(saliency_map[my, mx]))

            for px, py in pts[::2]:  # sample every second vertex
                vx = int(np.clip(px * sx, 0, w_map - 1))
                vy = int(np.clip(py * sy, 0, h_map - 1))
                sampled_values.append(float(saliency_map[vy, vx]))

            mean_saliency = float(np.mean(sampled_values))

            # Discretize into damage state
            if mean_saliency < 0.25:
                status = "no-damage"
                cat_key = "intact"
            elif mean_saliency < 0.50:
                status = "minor-damage"
                cat_key = "minor"
            elif mean_saliency < 0.75:
                status = "major-damage"
                cat_key = "major"
            else:
                status = "destroyed"
                cat_key = "destroyed"

            sec = b["sector"]
            sector_stats[sec][cat_key] += 1
            sector_stats[sec]["total"] += 1

            assessed_buildings.append({
                "id": b["id"],
                "sector": sec,
                "centroid": b["centroid"],
                "pts": pts,
                "change_intensity": round(mean_saliency, 3),
                "inferred_damage": status,
            })

        # Calculate overall disaster severity score (0 to 10)
        total_bld = len(self.osm_prior.buildings)
        if total_bld > 0:
            destroyed_total = sum(sector_stats[s]["destroyed"] for s in SECTOR_NAMES)
            major_total = sum(sector_stats[s]["major"] for s in SECTOR_NAMES)
            minor_total = sum(sector_stats[s]["minor"] for s in SECTOR_NAMES)

            score = ((destroyed_total * 1.0 + major_total * 0.7 + minor_total * 0.3) / total_bld) * 10.0
            score = round(min(10.0, score), 1)
        else:
            score = 0.0

        if score >= 6.5:
            priority = "CRITICAL (Immediate Search & Rescue Required)"
        elif score >= 4.0:
            priority = "HIGH (Substantial Structural Failures)"
        elif score >= 2.0:
            priority = "MODERATE (Localized Structural Impacts)"
        else:
            priority = "LOW (Minimal Structural Disruption)"

        # Find hotspot sector
        hotspot_sector = max(SECTOR_NAMES, key=lambda s: sector_stats[s]["destroyed"] * 2 + sector_stats[s]["major"])

        return {
            "total_structures": total_bld,
            "disaster_severity_score": score,
            "emergency_priority": priority,
            "hotspot_sector": hotspot_sector,
            "hotspot_sector_name": SECTOR_FULL_NAMES[hotspot_sector],
            "sector_breakdown": sector_stats,
            "buildings": assessed_buildings,
        }


def render_sector_grid(pil_image, sector_stats=None, line_color=(0, 229, 255, 180)):
    """
    Renders tactical quadrant crosshairs (NW, NE, SW, SE) and structure counts onto the image.
    """
    img = pil_image.convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mid_x, mid_y = w // 2, h // 2

    # Draw dashed or solid tactical sector dividers
    draw.line([(mid_x, 0), (mid_x, h)], fill=line_color, width=2)
    draw.line([(0, mid_y), (w, mid_y)], fill=line_color, width=2)

    # Sector quadrant badges
    badge_coords = {
        "NW": (20, 20),
        "NE": (mid_x + 20, 20),
        "SW": (20, mid_y + 20),
        "SE": (mid_x + 20, mid_y + 20),
    }

    for sec, (bx, by) in badge_coords.items():
        # Dark pill badge background
        draw.rectangle([bx, by, bx + 110, by + 46], fill=(18, 22, 30, 210), outline=line_color[:3] + (160,), width=1)
        draw.text((bx + 8, by + 4), f"Sector: {sec}", fill=(0, 229, 255))
        if sector_stats and sec in sector_stats:
            stats = sector_stats[sec]
            info = f"Bld: {stats.get('total', 0)} | Dest: {stats.get('destroyed', 0)}"
        else:
            info = "Quadrant Active"
        draw.text((bx + 8, by + 24), info, fill=(220, 220, 220))

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def render_saliency_heatmap_overlay(pil_image, saliency_map, alpha=0.45):
    """
    Blends the spatial change attribution heatmap onto the satellite image.
    """
    import matplotlib.pyplot as plt

    w, h = pil_image.size
    # Resize saliency map to image dimensions
    sal_img = Image.fromarray((saliency_map * 255).astype(np.uint8))
    sal_resized = sal_img.resize((w, h), Image.BICUBIC)
    sal_arr = np.array(sal_resized) / 255.0

    # Apply turbo colormap (perceptually uniform)
    cmap = plt.get_cmap("turbo")
    heat_rgba = cmap(sal_arr)  # float RGBA in [0, 1]
    heat_rgba_uint8 = (heat_rgba * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgba_uint8, "RGBA")

    # Set transparency
    heat_pil.putalpha(int(alpha * 255))

    base_rgba = pil_image.convert("RGBA")
    fused = Image.alpha_composite(base_rgba, heat_pil)
    return fused.convert("RGB")
