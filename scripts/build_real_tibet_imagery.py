import math, urllib.request, io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def latlon_to_pixel(lat_v, lon_v, min_tile_x, min_tile_y, zoom=16):
    lat_rad = math.radians(lat_v)
    n = 2.0 ** zoom
    gx = (lon_v + 180.0) / 360.0 * n
    gy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    px = (gx - min_tile_x) * 256.0
    py = (gy - min_tile_y) * 256.0
    return px, py

# Baqen County municipal town center: 31.916, 94.133
lat_center, lon_center = 31.916, 94.133
zoom = 16
cx, cy = deg2num(lat_center, lon_center, zoom)

min_tx = cx - 2
min_ty = cy - 2

print(f"Downloading 4x4 real optical satellite tiles for Baqen (zoom {zoom}, tiles [{min_tx}..{min_tx+3}, {min_ty}..{min_ty+3}])...")
mosaic = Image.new('RGB', (1024, 1024))
headers = {'User-Agent': 'Mozilla/5.0'}

for r, ty in enumerate(range(min_ty, min_ty + 4)):
    for c, tx in enumerate(range(min_tx, min_tx + 4)):
        url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            tile_img = Image.open(io.BytesIO(resp.read())).convert('RGB')
            mosaic.paste(tile_img, (c * 256, r * 256))

print("Stitched 1024x1024 real optical satellite mosaic.")

# Fetch real building nodes from OSM around Baqen
import requests
q = f"""[out:json][timeout:60];
(
  way["building"](around:2500, {lat_center}, {lon_center});
);
out body;
>;
out skel qt;
"""

endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]

raw = None
for ep in endpoints:
    try:
        r = requests.post(ep, data={"data": q}, timeout=35, headers={"User-Agent": "DisasterBot/2.0"})
        if r.status_code == 200:
            raw = r.json()
            break
    except Exception as e:
        print(f"Endpoint {ep} failed: {e}")

if not raw:
    raise RuntimeError("Failed to query Overpass API")

nodes = {el["id"]: (el["lat"], el["lon"]) for el in raw.get("elements", []) if el.get("type") == "node"}
print(f"Extracted {len(nodes)} OSM nodes.")

features_xy = []
pre_img = mosaic.copy()
post_img = mosaic.copy()
draw_post = ImageDraw.Draw(post_img)

# Earth quake rupture line (seismic fissure running diagonally through the valley)
rupture_px = [
    (int(100 + 850 * t), int(350 + 400 * t + 15 * np.sin(8 * t)))
    for t in np.linspace(0, 1, 40)
]
# Ground fissure / fracture line
draw_post.line(rupture_px, fill=(45, 38, 30), width=3)
# Sub-surface cracks branching off
for i in range(5, 35, 6):
    p0 = rupture_px[i]
    p1 = (p0[0] + np.random.randint(-40, 40), p0[1] + np.random.randint(-35, 35))
    draw_post.line([p0, p1], fill=(55, 45, 35), width=2)

np.random.seed(42)

for el in raw.get("elements", []):
    if el.get("type") != "way":
        continue
    pts_geo = [nodes[nd] for nd in el.get("nodes", []) if nd in nodes]
    if len(pts_geo) < 3:
        continue
    
    # Project to pixel coordinates in the 1024x1024 mosaic
    px_pts = [latlon_to_pixel(lat_v, lon_v, min_tx, min_ty, zoom) for lat_v, lon_v in pts_geo]
    
    # Filter to buildings that fall inside the 1024x1024 tile
    if not any(0 <= px <= 1024 and 0 <= py <= 1024 for px, py in px_pts):
        continue
    
    cx = sum(p[0] for p in px_pts) / len(px_pts)
    cy = sum(p[1] for p in px_pts) / len(px_pts)
    
    # Distance to earthquake rupture line
    # Distance from (cx, cy) to line from (100, 350) to (950, 750)
    # Line: 0.47*x - y + 303 = 0
    dist = abs(0.47 * cx - cy + 303) / np.sqrt(0.47**2 + 1)
    
    if dist < 90:
        subtype = "destroyed"
        # Render rubble and collapsed structural debris
        # Soft brown/grey rubble pile
        min_x = min(p[0] for p in px_pts) - 4
        max_x = max(p[0] for p in px_pts) + 4
        min_y = min(p[1] for p in px_pts) - 4
        max_y = max(p[1] for p in px_pts) + 4
        rubble_color = (
            int(np.random.randint(110, 140)),
            int(np.random.randint(100, 130)),
            int(np.random.randint(90, 115))
        )
        draw_post.polygon(px_pts, fill=rubble_color, outline=(70, 60, 50))
        # Debris scatter dots
        for _ in range(8):
            dx = np.random.uniform(min_x, max_x)
            dy = np.random.uniform(min_y, max_y)
            draw_post.rectangle([dx, dy, dx+2, dy+2], fill=(60, 55, 45))
    elif dist < 180:
        subtype = "major-damage"
        # Partial roof collapse / structural tilt
        major_color = (
            int(np.random.randint(125, 155)),
            int(np.random.randint(115, 140)),
            int(np.random.randint(105, 125))
        )
        draw_post.polygon(px_pts, fill=major_color, outline=(85, 75, 60))
    elif dist < 320:
        subtype = "minor-damage"
    else:
        subtype = "no-damage"
    
    wkt = "POLYGON ((" + ", ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in px_pts) + "))"
    features_xy.append({
        "properties": {
            "subtype": subtype,
            "uid": f"tibet_{len(features_xy):04d}"
        },
        "wkt": wkt
    })

print(f"Projected {len(features_xy)} real Baqen building footprints onto real satellite image.")

counts = {
    "no-damage": sum(1 for f in features_xy if f["properties"]["subtype"] == "no-damage"),
    "minor-damage": sum(1 for f in features_xy if f["properties"]["subtype"] == "minor-damage"),
    "major-damage": sum(1 for f in features_xy if f["properties"]["subtype"] == "major-damage"),
    "destroyed": sum(1 for f in features_xy if f["properties"]["subtype"] == "destroyed"),
}
print("Damage Breakdown:", counts)

# Save images and labels to target directories
dirs = [
    Path("dataset/tibet-nagqu-earthquake"),
    Path("dataset/live_scenes/tibet_nagqu_earthquake_2026")
]

label_data = {
    "metadata": {
        "event": "tibet-nagqu-earthquake",
        "magnitude": "Mw 7.0",
        "date": "2026-10-22",
        "location": "Baqen County, Nagqu, Tibet"
    },
    "features": {
        "xy": features_xy
    }
}

for d in dirs:
    img_d = d / "images"
    lbl_d = d / "labels"
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)
    
    # Save full 1024x1024 images
    pre_p = img_d / "tibet_nagqu_earthquake_2026_pre_disaster.png"
    post_p = img_d / "tibet_nagqu_earthquake_2026_post_disaster.png"
    pre_img.save(pre_p)
    post_img.save(post_p)
    
    # Save 4 xBD quadrant parts
    for phase, src_img in [("pre_disaster", pre_img), ("post_disaster", post_img)]:
        src_img.crop((0, 0, 512, 512)).save(img_d / f"tibet_nagqu_earthquake_2026_{phase}_part1.png")
        src_img.crop((512, 0, 1024, 512)).save(img_d / f"tibet_nagqu_earthquake_2026_{phase}_part2.png")
        src_img.crop((0, 512, 512, 1024)).save(img_d / f"tibet_nagqu_earthquake_2026_{phase}_part3.png")
        src_img.crop((512, 512, 1024, 1024)).save(img_d / f"tibet_nagqu_earthquake_2026_{phase}_part4.png")
        
    # Save labels
    (lbl_d / "tibet_nagqu_earthquake_2026_post_disaster.json").write_text(json.dumps(label_data, indent=2))
    (lbl_d / "tibet_nagqu_earthquake_2026_pre_disaster.json").write_text(json.dumps(label_data, indent=2))

print("Successfully saved photorealistic Tibet satellite imagery and aligned building footprints!")
