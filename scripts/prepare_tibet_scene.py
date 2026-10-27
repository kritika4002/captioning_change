import sys
sys.path.insert(0, ".")
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

out_dir = Path("dataset/live_scenes/tibet_nagqu_earthquake_2026")
img_dir = out_dir / "images"
lbl_dir = out_dir / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

osm_file = out_dir / "osm_prior.geojson"
with open(osm_file, "r", encoding="utf-8") as f:
    osm_data = json.load(f)

features = osm_data.get("features", {}).get("xy", [])
print(f"Loaded {len(features)} Baqen building footprints from OSM.")

# Parse polygons
from src.data.damage_overlay import _parse_wkt_polygon

# Create realistic high-altitude Tibetan plateau satellite imagery (1024x1024)
# Plateau terrain palette: earthy ochre, mountain shadows, Tibetan valley green, river bed
np.random.seed(42)

# Base plateau satellite background
w, h = 1024, 1024
x = np.linspace(0, 1, w)
y = np.linspace(0, 1, h)
xx, yy = np.meshgrid(x, y)

# Elevation / mountain valley texture
elevation = (
    0.4 * np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy)
    + 0.3 * np.sin(7 * np.pi * xx + 2 * np.pi * yy)
    + 0.2 * np.random.normal(0, 0.05, (h, w))
)
elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

# Colors for high-altitude Tibetan valley (Baqen: elevation 4,100m, alpine meadow + river valley)
pre_rgb = np.zeros((h, w, 3), dtype=np.uint8)
# Base Tibetan plateau rock/meadow
pre_rgb[:, :, 0] = np.clip(135 + 40 * elevation + np.random.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)  # R
pre_rgb[:, :, 1] = np.clip(130 + 35 * elevation + np.random.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)  # G
pre_rgb[:, :, 2] = np.clip(105 + 30 * elevation + np.random.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)  # B

# River meandering through Baqen town
river_mask = np.abs(yy - (0.45 + 0.15 * np.sin(4 * np.pi * xx) + 0.05 * np.cos(8 * np.pi * xx))) < 0.025
pre_rgb[river_mask] = [75, 95, 115]  # Cold glacial river tone

pre_img = Image.fromarray(pre_rgb).filter(ImageFilter.GaussianBlur(radius=0.8))
draw_pre = ImageDraw.Draw(pre_img)

# Draw intact Tibetan stone/tin-roofed buildings in pre-disaster image
for feat in features:
    pts = _parse_wkt_polygon(feat.get("wkt", ""))
    if len(pts) >= 3:
        # Tibetan roofs: vibrant blue, crimson red, or silver tin
        roof_color = (
            int(np.random.choice([165, 180, 50, 130])),
            int(np.random.choice([50, 70, 80, 125])),
            int(np.random.choice([45, 60, 140, 120])),
        )
        draw_pre.polygon(pts, fill=roof_color, outline=(40, 35, 30))

pre_img_path = img_dir / "tibet_nagqu_earthquake_2026_pre_disaster.png"
pre_img.save(pre_img_path)
print(f"Saved pre-disaster image: {pre_img_path}")

# Post-disaster: earthquake ground rupture, surface fissures, debris and collapsed buildings
post_img = pre_img.copy()
draw_post = ImageDraw.Draw(post_img)

# Surface earthquake strike-slip rupture zone running NW to SE
rupture_pts = [(int(50 + 924 * t), int(300 + 400 * t + 25 * np.sin(12 * t))) for t in np.linspace(0, 1, 35)]
draw_post.line(rupture_pts, fill=(60, 50, 40), width=4)

# Structural damage states for Baqen buildings:
# Near fault rupture: destroyed & major damage; further away: minor & intact
post_features = []
pre_features = []

for idx, feat in enumerate(features):
    pts = _parse_wkt_polygon(feat.get("wkt", ""))
    if len(pts) < 3:
        continue
    
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    
    # Distance to earthquake rupture line
    # Distance from (cx, cy) to line from (50, 300) to (974, 700)
    # y - 0.43*x - 278 = 0
    dist = abs(cy - 0.43 * cx - 278) / np.sqrt(1 + 0.43**2)
    
    if dist < 85:
        subtype = "destroyed"
        # Overwrite building with earthquake rubble / collapsed debris
        debris_color = (110, 100, 85)
        draw_post.polygon(pts, fill=debris_color, outline=(70, 60, 50))
        # Draw rubble scatter
        for _ in range(5):
            rx = cx + np.random.normal(0, 6)
            ry = cy + np.random.normal(0, 6)
            draw_post.ellipse([rx - 2, ry - 2, rx + 2, ry + 2], fill=(130, 120, 100))
    elif dist < 175:
        subtype = "major-damage"
        # Partially collapsed roof
        draw_post.polygon(pts, fill=(135, 95, 75), outline=(65, 55, 45))
    elif dist < 280:
        subtype = "minor-damage"
        # Minor cracking
        draw_post.polygon(pts, fill=(150, 120, 95), outline=(55, 50, 45))
    else:
        subtype = "no-damage"

    post_features.append({
        "properties": {
            "subtype": subtype,
            "uid": f"tibet_{idx:04d}",
        },
        "wkt": feat.get("wkt", ""),
    })
    
    pre_features.append({
        "properties": {
            "subtype": "no-damage",
            "uid": f"tibet_{idx:04d}",
        },
        "wkt": feat.get("wkt", ""),
    })

post_img_path = img_dir / "tibet_nagqu_earthquake_2026_post_disaster.png"
post_img.save(post_img_path)
print(f"Saved post-disaster image: {post_img_path}")

# Save matching post and pre labels
post_lbl_file = lbl_dir / "tibet_nagqu_earthquake_2026_post_disaster.json"
post_lbl_file.write_text(json.dumps({
    "metadata": {
        "event": "tibet-nagqu-earthquake",
        "magnitude": "Mw 7.0",
        "date": "2026-10-22",
        "location": "Baqen County, Nagqu, Tibet",
    },
    "features": {
        "xy": post_features,
    }
}, indent=2))

pre_lbl_file = lbl_dir / "tibet_nagqu_earthquake_2026_pre_disaster.json"
pre_lbl_file.write_text(json.dumps({
    "metadata": {
        "event": "tibet-nagqu-earthquake",
        "date": "2026-10-07",
    },
    "features": {
        "xy": pre_features,
    }
}, indent=2))

print(f"Saved post label with {len(post_features)} features to {post_lbl_file}")
print("Tibet Nagqu live scene preparation complete!")
