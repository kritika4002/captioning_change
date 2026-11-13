
import os
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise SystemExit("Install Pillow and numpy: pip install Pillow numpy")


# ── Copernicus credentials (load from .env or env vars) ─────────────────────
def _load_creds():
    cid = os.getenv("COPERNICUS_CLIENT_ID")
    csec = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not cid or not csec:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("COPERNICUS_CLIENT_ID="):
                    cid = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("COPERNICUS_CLIENT_SECRET="):
                    csec = line.split("=", 1)[1].strip().strip('"')

    if not cid or not csec:
        raise SystemExit(
            "Missing Copernicus credentials.\n"
            "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in your .env file."
        )
    return cid, csec


# ── OAuth token ──────────────────────────────────────────────────────────────
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

def get_token(client_id, client_secret):
    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Sentinel-2 catalogue search ──────────────────────────────────────────────
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

def _bbox(lat, lon, delta=0.05):
    """Returns a bounding box around a point."""
    return (lon - delta, lat - delta, lon + delta, lat + delta)


def search_sentinel2(token, lat, lon, date_from, date_to, max_cloud=30, limit=5):
    """Search Copernicus Data Space for Sentinel-2 L2A scenes."""
    w, s, e, n = _bbox(lat, lon)
    aoi = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud}) and "
            f"ContentDate/Start gt {date_from}T00:00:00.000Z and "
            f"ContentDate/Start lt {date_to}T23:59:59.999Z and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": limit,
    }

    resp = requests.get(
        CATALOG_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


# ── Download helper ──────────────────────────────────────────────────────────
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"

def download_product(token, product_id, out_path: Path, chunk_mb=8):
    """Streams a product zip from CDSE to disk."""
    url = DOWNLOAD_URL.format(id=product_id)
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            fh.write(chunk)
            written += len(chunk)
    print(f"  -> Downloaded {written / 1e6:.1f} MB -> {out_path}")
    return out_path


# ── Quick RGB preview from downloaded zip ────────────────────────────────────
def extract_rgb_preview(zip_path: Path, out_png: Path):
    """Extracts the JPEG2000 TCI (True-Color Image) from a Sentinel-2 zip."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            tci_files = [n for n in zf.namelist() if "TCI_10m" in n and n.endswith(".jp2")]
            if not tci_files:
                tci_files = [n for n in zf.namelist() if "TCI" in n]
            if not tci_files:
                print("  ⚠ No TCI file found in zip -- skipping preview")
                return None
            tci_name = tci_files[0]
            raw = zf.read(tci_name)

        from io import BytesIO
        try:
            import rasterio
            from rasterio.io import MemoryFile
            with MemoryFile(raw) as mf:
                with mf.open() as ds:
                    data = ds.read([1, 2, 3])  # RGB
                    img = Image.fromarray(np.moveaxis(data, 0, -1).astype(np.uint8))
        except ImportError:
            # Fallback: save raw jp2 as-is
            out_png = out_png.with_suffix(".jp2")
            out_png.write_bytes(raw)
            print(f"  -> Raw TCI saved (install rasterio for PNG preview): {out_png}")
            return out_png

        img = img.resize((1024, 1024), Image.LANCZOS)
        img.save(out_png)
        print(f"  -> RGB preview: {out_png}")
        return out_png

    except Exception as ex:
        print(f"  ⚠ Preview extraction failed: {ex}")
        return None


# ── OSM Overpass -- live building footprints ──────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_osm_buildings(lat, lon, radius_m=1000, out_json: Path = None):
    """
    Queries the Overpass API for building footprints within `radius_m` metres
    of (lat, lon) and saves them in a GeoJSON-compatible format readable by
    OSMBuildingPrior.
    """
    query = f"""
    [out:json][timeout:60];
    (
      way["building"](around:{radius_m},{lat},{lon});
      relation["building"](around:{radius_m},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    print(f"Fetching OSM buildings (r={radius_m}m around {lat:.4f},{lon:.4f})...")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    raw = None
    for ep in endpoints:
        try:
            resp = requests.post(
                ep,
                data={"data": query},
                headers={"User-Agent": "DisasterAssessmentBot/1.0 (disaster-relief@edu)"},
                timeout=45,
            )
            if resp.status_code == 200:
                raw = resp.json()
                break
        except Exception:
            continue
    if raw is None or "elements" not in raw:
        print("  [!] Overpass server timed out or returned no data. Returning empty building prior collection.")
        empty_geojson = {"type": "FeatureCollection", "features": []}
        if out_json:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(empty_geojson, indent=2))
        return empty_geojson, 0

    # Build node lookup
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in raw["elements"] if el["type"] == "node"}

    features = []
    for el in raw["elements"]:
        if el["type"] != "way":
            continue
        pts = [nodes[nd] for nd in el.get("nodes", []) if nd in nodes]
        if len(pts) < 3:
            continue
        # Convert geographic coords -> approximate pixel coords (1024x1024 canvas)
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)

        def geo_to_px(lon_v, lat_v, lon_ref, lat_ref, scale=5000):
            x = (lon_v - lon_ref) * scale + 512
            y = (lat_ref - lat_v) * scale + 512  # y inverted (north up)
            return x, y

        px_pts = [geo_to_px(p[0], p[1], lon, lat) for p in pts]
        wkt = "POLYGON ((" + ", ".join(f"{x:.2f} {y:.2f}" for x, y in px_pts) + "))"

        features.append({
            "osm_id": el["id"],
            "wkt": wkt,
            "tags": el.get("tags", {}),
        })

    geojson = {"type": "FeatureCollection", "features": {"xy": features}}
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(geojson, indent=2))
        print(f"  -> {len(features)} building footprints saved -> {out_json}")
    return geojson, len(features)


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch live Copernicus + OSM scene")
    parser.add_argument("--lat",       type=float, required=True,  help="Scene centre latitude")
    parser.add_argument("--lon",       type=float, required=True,  help="Scene centre longitude")
    parser.add_argument("--event",     type=str,   required=True,  help="Disaster event name (used as folder name)")
    parser.add_argument("--pre-date",  type=str,   required=True,  help="Pre-disaster date (YYYY-MM-DD)")
    parser.add_argument("--post-date", type=str,   required=True,  help="Post-disaster date (YYYY-MM-DD)")
    parser.add_argument("--out-dir",   type=str,   default="dataset/live_scenes")
    parser.add_argument("--cloud",     type=int,   default=30,     help="Max cloud cover %")
    parser.add_argument("--osm-radius",type=int,   default=1500,   help="OSM building search radius (metres)")
    parser.add_argument("--no-download", action="store_true",      help="Skip full zip download (preview only)")
    args = parser.parse_args()

    cid, csec = _load_creds()
    out_root = Path(args.out_dir) / args.event
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Live Scene Fetch: {args.event}")
    print(f"Location : {args.lat:.4f}°N, {args.lon:.4f}°E")
    print(f"Pre  window: {args.pre_date}")
    print(f"Post window: {args.post_date}")
    print(f"{'='*60}\n")

    # 1. Get OAuth token
    print("Authenticating with Copernicus Data Space...")
    token = get_token(cid, csec)
    print("  -> Token acquired ✓")

    # 2. Search pre-disaster scene
    pre_end = (datetime.strptime(args.pre_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"\nSearching pre-disaster Sentinel-2 scenes ({args.pre_date} -> {pre_end})...")
    pre_products = search_sentinel2(token, args.lat, args.lon, args.pre_date, pre_end, args.cloud)
    if not pre_products:
        print("  ✗ No pre-disaster scenes found. Try widening date range or increasing --cloud.")
        return
    pre_prod = pre_products[0]
    print(f"  -> Found: {pre_prod['Name']} ({pre_prod.get('ContentDate',{}).get('Start','?')[:10]})")

    # 3. Search post-disaster scene
    post_end = (datetime.strptime(args.post_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"\nSearching post-disaster Sentinel-2 scenes ({args.post_date} -> {post_end})...")
    post_products = search_sentinel2(token, args.lat, args.lon, args.post_date, post_end, args.cloud)
    if not post_products:
        print("  ✗ No post-disaster scenes found. Try widening date range or increasing --cloud.")
        return
    post_prod = post_products[0]
    print(f"  -> Found: {post_prod['Name']} ({post_prod.get('ContentDate',{}).get('Start','?')[:10]})")

    # 4. Download or just get previews
    scene_id = f"{args.event}_live"

    if not args.no_download:
        print("\nDownloading pre-disaster scene (this may take several minutes)...")
        pre_zip = download_product(token, pre_prod["Id"], out_root / f"{scene_id}_pre.zip")
        print("Downloading post-disaster scene...")
        post_zip = download_product(token, post_prod["Id"], out_root / f"{scene_id}_post.zip")
        print("\nExtracting RGB previews...")
        pre_png  = extract_rgb_preview(pre_zip,  out_root / f"{scene_id}_pre_disaster.png")
        post_png = extract_rgb_preview(post_zip, out_root / f"{scene_id}_post_disaster.png")
    else:
        print("\n[--no-download] Skipping full product download. Only metadata saved.")
        pre_png = post_png = None

    # 5. OSM live building footprints
    print("\nFetching live OSM building footprints...")
    osm_out = out_root / f"{scene_id}_osm_prior.json"
    _, n_blds = fetch_osm_buildings(args.lat, args.lon, args.osm_radius, osm_out)

    # 6. Write scene manifest
    manifest = {
        "event":          args.event,
        "sample_id":      scene_id,
        "lat":            args.lat,
        "lon":            args.lon,
        "images_dir":     str(out_root),
        "pre_product":    pre_prod["Name"],
        "post_product":   post_prod["Name"],
        "pre_image":      str(pre_png)  if pre_png  else None,
        "post_image":     str(post_png) if post_png else None,
        "osm_prior_json": str(osm_out),
        "osm_buildings":  n_blds,
        "fetched_at":     datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = out_root / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print("FETCH COMPLETE")
    print(f"  Pre-disaster image  : {pre_png or 'not downloaded'}")
    print(f"  Post-disaster image : {post_png or 'not downloaded'}")
    print(f"  OSM buildings       : {n_blds}")
    print(f"  Manifest            : {manifest_path}")
    print(f"{'='*60}\n")
    print("Next step -- run inference on this live scene:")
    print(f"  python scripts/demo_inference.py --scene {manifest_path}\n")


if __name__ == "__main__":
    main()
