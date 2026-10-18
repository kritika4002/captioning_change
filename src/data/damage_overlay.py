"""
Damage Polygon Outline Renderer


Reads xBD JSON label files and draws color-coded building polygon outlines
directly onto PIL Images. Supports both WKT polygon format and GeoJSON coordinates.
This makes damage assessment visually obvious for human reviewers — each building
is outlined in a color that encodes its structural damage level.

Color scheme:
  No Damage    -> green   (#00e676)
  Minor Damage -> yellow  (#ffeb3b)
  Major Damage -> orange  (#ff9800)
  Destroyed    -> red     (#f44336)
  Unknown      -> gray    (#9e9e9e)
"""

import json
import re
from pathlib import Path
from PIL import Image, ImageDraw

DAMAGE_COLORS = {
    "no-damage":     (0,   230, 118, 200),   # bright green
    "minor-damage":  (255, 235,  59, 200),   # amber yellow
    "major-damage":  (255, 152,   0, 200),   # orange
    "destroyed":     (244,  67,  54, 220),   # red
    "un-classified": (158, 158, 158, 160),   # gray
}
DEFAULT_COLOR = (200, 200, 200, 160)
OUTLINE_WIDTH = 2


def _parse_wkt_polygon(wkt_str):
    """
    Extract coordinate pairs from a WKT polygon string like:
    POLYGON ((532.18 165.98, 431.27 201.22, ...))
    """
    if not wkt_str:
        return []
    
    # Strip POLYGON and nested parentheses
    clean = re.sub(r"[A-Za-z]+|\(|\)", "", wkt_str).strip()
    pts = []
    for pair in clean.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


def _extract_polygons_from_label(label_path):
    """
    Parse an xBD JSON label file and return list of (pts, subtype, uid, centroid) tuples.
    Handles both xBD 'wkt' format and GeoJSON 'geometry.coordinates'.
    """
    path = Path(label_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    polygons = []
    features = data.get("features", {})
    xy_features = features.get("xy", [])

    for feat in xy_features:
        pts = []
        # Case 1: WKT string (standard in xBD dataset)
        if "wkt" in feat:
            pts = _parse_wkt_polygon(feat.get("wkt", ""))
        
        # Case 2: GeoJSON geometry dictionary
        elif "geometry" in feat:
            coords = feat["geometry"].get("coordinates", [])
            if coords and len(coords[0]) >= 3:
                pts = [(float(c[0]), float(c[1])) for c in coords[0] if len(c) >= 2]

        if len(pts) < 3:
            continue

        props = feat.get("properties", {})
        subtype = props.get("subtype", "un-classified")
        uid = props.get("uid", "")

        # Compute simple centroid
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        polygons.append((pts, subtype, uid, (cx, cy)))

    return polygons


def draw_damage_outlines(pil_image, label_path, alpha_blend=True, scale_to_image=True):
    """
    Draw color-coded building polygon outlines onto pil_image.

    Parameters
    ----------
    pil_image   : PIL.Image (RGB)
    label_path  : str or Path — path to xBD JSON label file
    alpha_blend : bool — if True, use semi-transparent fills
    scale_to_image : bool — scales coordinates if image was resized from 1024x1024

    Returns
    -------
    PIL.Image (RGB) with outlines drawn
    """
    polygons = _extract_polygons_from_label(label_path)
    if not polygons:
        return pil_image

    img = pil_image.convert("RGBA")
    w, h = img.size

    # xBD canonical resolution is 1024x1024
    sx = w / 1024.0 if scale_to_image else 1.0
    sy = h / 1024.0 if scale_to_image else 1.0

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_over = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(img)

    for item in polygons:
        pts = item[0]
        subtype = item[1]
        
        # Scale coordinates if image size differs
        scaled_pts = [(p[0] * sx, p[1] * sy) for p in pts]
        
        color = DAMAGE_COLORS.get(subtype, DEFAULT_COLOR)
        fill_color = color[:3] + (55,)       # soft translucent fill
        outline_color = color[:3] + (230,)   # distinct vibrant contour

        try:
            if alpha_blend:
                draw_over.polygon(scaled_pts, fill=fill_color)
            draw_main.line(scaled_pts + [scaled_pts[0]], fill=outline_color, width=OUTLINE_WIDTH)
        except Exception:
            pass

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def get_damage_summary_from_label(label_path):
    """
    Returns damage count dictionary and full parsed building list from an xBD label file.
    """
    polygons = _extract_polygons_from_label(label_path)
    counts = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "destroyed": 0,
        "un-classified": 0,
    }
    for item in polygons:
        subtype = item[1]
        counts[subtype] = counts.get(subtype, 0) + 1
    return counts, polygons


def draw_legend(pil_image, position="bottom-right", box_size=14, padding=10, font_size=12):
    """
    Draws a compact damage color legend onto pil_image.
    """
    img = pil_image.convert("RGB")
    draw = ImageDraw.Draw(img)

    labels = [
        ("No Damage",    DAMAGE_COLORS["no-damage"][:3]),
        ("Minor Damage", DAMAGE_COLORS["minor-damage"][:3]),
        ("Major Damage", DAMAGE_COLORS["major-damage"][:3]),
        ("Destroyed",    DAMAGE_COLORS["destroyed"][:3]),
    ]

    line_h = box_size + 5
    total_h = len(labels) * line_h + 2 * padding
    total_w = 125 + 2 * padding

    W, H = img.size
    if position == "bottom-right":
        x0 = W - total_w - padding
        y0 = H - total_h - padding
    elif position == "top-right":
        x0, y0 = W - total_w - padding, padding
    else:
        x0, y0 = padding, H - total_h - padding

    # Semi-transparent dark background card
    draw.rectangle([x0 - 2, y0 - 2, x0 + total_w + 2, y0 + total_h + 2],
                   fill=(24, 28, 36))

    for i, (label, color) in enumerate(labels):
        cy = y0 + padding + i * line_h
        draw.rectangle([x0 + padding, cy, x0 + padding + box_size, cy + box_size], fill=color)
        draw.text((x0 + padding + box_size + 8, cy), label, fill=(240, 240, 240))

    return img
