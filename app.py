from flask import Flask, request, jsonify, render_template
import geopandas as gpd
import pandas as pd
import numpy as np
import os

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Data paths (WORKS LOCALLY + ON RENDER)
# -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

BASE_DATA = os.path.join(HERE, "data", "la_erosion")

PRED_PATH = os.path.join(BASE_DATA, "la_erosion_predictions.geojson")
ZIP_PATH = os.path.join(BASE_DATA, "la_zip_centroids.csv")

print("BASE_DATA:", BASE_DATA)
print("PRED_PATH:", PRED_PATH, "exists?", os.path.exists(PRED_PATH))
print("ZIP_PATH:", ZIP_PATH, "exists?", os.path.exists(ZIP_PATH))

# -------------------------------------------------
# Load erosion predictions
# -------------------------------------------------
print("📌 Loading prediction dataset...")
gdf = gpd.read_file(PRED_PATH)

# --- FIX is_land coming in wrong from GeoJSON ---
if "is_land" in gdf.columns:
    s = gdf["is_land"]

    # If it's already bool, keep it
    if s.dtype != "bool":
        # Normalize strings/numbers -> bool
        s_str = s.astype(str).str.strip().str.lower()
        gdf["is_land"] = s_str.isin(["true", "1", "t", "yes", "y"])
    else:
        gdf["is_land"] = s.astype(bool)

print("Land cells in Flask (after fix):", int(gdf["is_land"].sum()))
print("Sample is_land values:", gdf["is_land"].head(10).tolist())

print("✅ GeoJSON loaded from:", PRED_PATH)
print("✅ Rows:", len(gdf), "Columns:", list(gdf.columns))
print("✅ erosion_proba describe:", gdf["erosion_proba"].describe() if "erosion_proba" in gdf.columns else "MISSING")
print("✅ risk_level counts:", gdf["risk_level"].value_counts().head(10) if "risk_level" in gdf.columns else "MISSING")
print("Land cells in Flask:", gdf["is_land"].sum())

# Ensure lat/lon exist
if "lat" not in gdf.columns or "lon" not in gdf.columns:
    centroids = gdf.geometry.centroid
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

print("✅ Loaded rows:", len(gdf))
print("✅ Columns:", list(gdf.columns))

# Strip geometry for fast lookup
df = pd.DataFrame(gdf.drop(columns="geometry"))
coords = df[["lat", "lon"]].to_numpy()

# ---------------------------------------------------
# Land filter (prevents water-only cells from being used)
# ---------------------------------------------------
# A cell is "land-relevant" if it had ANY land-loss/gain signal,
# or is labeled as land change, or has non-trivial loss/gain fraction.
LAND_FRAC_EPS = 0.0001  # tweak if needed

df["is_land_cell"] = (
    (df.get("loss_frac", 0.0) > LAND_FRAC_EPS) |
    (df.get("gain_frac", 0.0) > LAND_FRAC_EPS) |
    (df.get("loss_label", 0) == 1) |
    (df.get("gain_label", 0) == 1)
)

# -------------------------------------------------
# Load ZIP centroids
# -------------------------------------------------
if os.path.exists(ZIP_PATH):
    zip_df = pd.read_csv(ZIP_PATH, dtype={"zip": str})
    zip_df["zip"] = zip_df["zip"].str.zfill(5)
    print("✅ Loaded ZIP centroids:", len(zip_df))
else:
    zip_df = None
    print("⚠️ ZIP file not found")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
import numpy as np

# Build once after loading gdf
# (Put this right after you load gdf in app.py)
coords = np.column_stack([gdf["lat"].to_numpy(), gdf["lon"].to_numpy()])
land_mask = gdf.get("is_land", False)
if not isinstance(land_mask, np.ndarray):
    land_mask = gdf["is_land"].to_numpy()

def _haversine_m(lat1, lon1, lat2, lon2):
    # lat/lon in degrees, returns meters
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def find_nearest_row(lat: float, lon: float):
    dlat = coords[:, 0] - lat
    dlon = coords[:, 1] - lon
    dist2 = dlat * dlat + dlon * dlon

    # prefer land cells if available
    if "is_land" in df.columns:
        land_mask = df["is_land"].astype(bool).to_numpy()
        if land_mask.any():
            dist2_land = dist2.copy()
            dist2_land[~land_mask] = float("inf")
            idx = int(dist2_land.argmin())
            if dist2_land[idx] != float("inf"):
                return df.iloc[idx]

    # fallback
    idx = int(dist2.argmin())
    return df.iloc[idx]

def row_to_payload(row, input_lat=None, input_lon=None, input_zip=None):
    START_YEAR = 1990
    END_YEAR = 2016
    YEARS = END_YEAR - START_YEAR  # 26

    loss_frac = float(row.get("loss_frac", 0.0) or 0.0)
    gain_frac = float(row.get("gain_frac", 0.0) or 0.0)

    erosion_proba = float(row.get("erosion_proba", 0.0) or 0.0)

    # --- recompute 3-tier risk level (NO very_low) ---
    if erosion_proba < 0.15:
        risk_level = "low"
    elif erosion_proba < 0.40:
        risk_level = "moderate"
    else:
        risk_level = "high"

    erosion_pred = 1 if erosion_proba >= 0.40 else 0

    # --- make pred consistent with buckets ---
    erosion_pred = 1 if erosion_proba >= 0.25 else 0

    annual_loss_rate = (loss_frac / YEARS) if YEARS > 0 else 0.0

    def years_to_loss(target):
        if annual_loss_rate > 0:
            years = target / annual_loss_rate
            # ignore insane timelines (prevents 0 months + nonsense)
            if years <= 0 or years > 200:
                return None
            return years
        return None

    payload = {
        "cell_lat": float(row["lat"]),
        "cell_lon": float(row["lon"]),

        "loss_frac": loss_frac,
        "gain_frac": gain_frac,

        "erosion_proba": erosion_proba,
        "erosion_pred": erosion_pred,
        "risk_level": risk_level,

        "period_start": START_YEAR,
        "period_end": END_YEAR,
        "period_years": YEARS,
        "obs_period": f"{START_YEAR}–{END_YEAR}",
        "obs_years": YEARS,

        "annual_loss_rate": annual_loss_rate,
        "years_to_1pct_loss": years_to_loss(0.01),
        "years_to_5pct_loss": years_to_loss(0.05),
        "years_to_10pct_loss": years_to_loss(0.10),
        "years_to_major_change": years_to_loss(0.10),

        "input_lat": input_lat,
        "input_lon": input_lon,
        "input_zip": input_zip,
    }

    return payload

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/preds")
def preds():
    # This returns the GeoJSON file to the browser 
    return send_from_directory(BASE_DATA, "la_erosion_predictions.geojson")

@app.route("/api/erosion")
def erosion_point():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except:
        return jsonify({"error": "Invalid lat/lon"}), 400

    row = find_nearest_row(lat, lon)

    # --- robust land check (supports is_land or is_land_cell, bool or string) ---
    land_val = None
    if "is_land" in row:
        land_val = row.get("is_land")
    elif "is_land_cell" in row:
        land_val = row.get("is_land_cell")

    is_land = False
    if isinstance(land_val, (bool, int, float)):
        is_land = bool(land_val)
    elif land_val is not None:
        is_land = str(land_val).strip().lower() in ["true", "1", "t", "yes", "y"]

    if not is_land:
        return jsonify({"water": True, "message": "This location appears to be open water"}), 200

    return jsonify(row_to_payload(row, input_lat=lat, input_lon=lon, input_zip=None))

@app.route("/api/erosion_zip")
def erosion_zip():
    if zip_df is None:
        return jsonify({"error": "ZIP data unavailable"}), 500

    z = request.args.get("zip", "").zfill(5)
    match = zip_df[zip_df["zip"] == z]

    if match.empty:
        return jsonify({"error": f"ZIP {z} not found"}), 404

    lat = float(match.iloc[0]["lat"])
    lon = float(match.iloc[0]["lon"])

    row = find_nearest_row(lat, lon)

    land_val = None
    if "is_land" in row:
        land_val = row.get("is_land")
    elif "is_land_cell" in row:
        land_val = row.get("is_land_cell")

    is_land = False
    if isinstance(land_val, (bool, int, float)):
        is_land = bool(land_val)
    elif land_val is not None:
        is_land = str(land_val).strip().lower() in ["true", "1", "t", "yes", "y"]

    if not is_land:
        return jsonify({"water": True, "message": "ZIP centroid is open water"}), 200

    return jsonify(row_to_payload(row, input_lat=lat, input_lon=lon, input_zip=z))

@app.route("/api/debug_land")
def debug_land():
    cols = list(gdf.columns)
    # try to compute land counts safely
    land_col = "is_land" if "is_land" in gdf.columns else ("is_land_cell" if "is_land_cell" in gdf.columns else None)
    if land_col is None:
        return jsonify({"error": "No land column found", "columns": cols})

    s = gdf[land_col]
    # count "truthy" values the safe way
    land_true = int((s == True).sum()) if s.dtype != object else int((s.astype(str).str.lower().isin(["true","1","t","yes"])).sum())

    return jsonify({
        "rows": int(len(gdf)),
        "land_col": land_col,
        "dtype": str(s.dtype),
        "true_count": land_true,
        "sample_values": list(s.head(10).astype(str))
    })

from flask import send_file

@app.route("/data/preds")
def preds_file():
    return send_file(PRED_PATH, mimetype="application/geo+json")
# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
