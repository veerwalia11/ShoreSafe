// static/app.js

// ----------------------------
// 1) Init map
// ----------------------------
const map = L.map("map").setView([29.7, -90.2], 7);
setTimeout(() => map.invalidateSize(), 200);
window.addEventListener("resize", () => map.invalidateSize());

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>',
}).addTo(map);

let marker = L.marker([29.7, -90.2]).addTo(map);

const resultDiv = document.getElementById("result");
const zipForm = document.getElementById("zip-form");
const zipInput = document.getElementById("zip-input");

// ----------------------------
// Helpers
// ----------------------------
function normLevel(level) {
  if (!level) return "low";
  const s = String(level).trim().toLowerCase();
  if (s.includes("high")) return "high";
  if (s.includes("mod")) return "moderate";
  return "low";
}

function riskColor(level) {
  const l = normLevel(level);
  if (l === "high") return "#d73027";      // red
  if (l === "moderate") return "#fc8d59";  // orange
  return "#1a9850";                        // green
}

function fmtPct(x, digits = 1) {
  const n = Number(x);
  if (!isFinite(n)) return "—";
  return (n * 100).toFixed(digits);
}

function fmtNum(x, digits = 1) {
  const n = Number(x);
  if (!isFinite(n)) return "—";
  return n.toFixed(digits);
}

// Turn a single-year estimate into a friendlier range
function yearsRange(years) {
  const y = Number(years);
  if (!isFinite(y) || y <= 0) return null;

  // +/- 20% range, min span of 2 years
  const low = Math.max(0, Math.floor(y * 0.8));
  const high = Math.max(low + 2, Math.ceil(y * 1.2));

  // If it’s huge, bucket it
  if (high >= 120) return "100+ years";
  if (high >= 60) return "60–100 years";
  if (high >= 35) return "35–60 years";
  if (high >= 25) return "25–35 years";
  if (high >= 15) return "15–25 years";
  if (high >= 8) return "8–15 years";
  if (high >= 4) return "4–8 years";
  return "0–4 years";
}

function buildWhenBlock(data) {
  // Prefer a "major change" estimate if you have it
  const yMajor = yearsRange(data.years_to_major_change);
  const y10 = yearsRange(data.years_to_10pct_loss);

  const period = data.obs_period || "1990–2016";

  const chosen = yMajor || y10;
  if (!chosen) {
    return `
      <p><strong>Estimated timeline:</strong> Not available</p>
      <p style="font-size:12px;opacity:0.75;">
        This grid cell shows little/no observed loss during ${period}.
      </p>
    `;
  }

  return `
    <p><strong>Estimated timeline to major change:</strong> ${chosen}</p>
    <p style="font-size:12px;opacity:0.75;">
      Based on observed land change during ${period}.
    </p>
  `;
}

function riskColor(level) {
  if (level === "high") return "#d73027";
  if (level === "moderate") return "#fc8d59";
  return "#1a9850"; // low
}

function renderResult(data, sourceLabel) {
  // 1) Basic error handling
  if (!data || data.error) {
    resultDiv.innerHTML = `<p style="color:#b91c1c;"><strong>Error:</strong> ${data?.error || "Unknown error"}</p>`;
    return;
  }

  // 2) Water handling
  if (data.water) {
    resultDiv.innerHTML = `
      <p><strong>Source:</strong> ${sourceLabel}</p>
      <p style="color:#374151;"><strong>Result:</strong> This location is open water.</p>
    `;
    return;
  }

  // 3) Risk + color
  const level = (data.risk_level || "low").toLowerCase();
  const color = riskColor(level);

  // 4) Probability-based timeline (works even when loss_frac = 0)
  const p = Number(data.erosion_proba);
  let timeline;

  if (!isFinite(p) || p <= 0) {
    timeline = "Low near-term erosion risk";
  } else if (p >= 0.75) {
    timeline = "Likely within 5–8 years";
  } else if (p >= 0.50) {
    timeline = "Likely within 8–15 years";
  } else if (p >= 0.25) {
    timeline = "Possible within 15–30 years";
  } else {
    timeline = "Low near-term erosion risk";
  }

  // 5) Render sidebar
  resultDiv.innerHTML = `
    <p><strong>Source:</strong> ${sourceLabel}</p>

    <p>
      <strong>Erosion risk:</strong>
      <span style="
        display:inline-block;
        padding:2px 6px;
        border-radius:4px;
        background:${color};
        color:white;
        font-weight:600;">
        ${level}
      </span>
    </p>

    <p><strong>Chance of erosion risk (model):</strong> ${fmtPct(p)}%</p>

    <p><strong>Estimated time to erosion:</strong> ${timeline}</p>

    <hr style="border:none;border-top:1px solid #e5e7eb;margin:10px 0;" />

    <p style="font-size:12px;opacity:0.8;">
      <strong>Nearest grid cell center:</strong><br/>
      (${Number(data.cell_lat).toFixed(6)}, ${Number(data.cell_lon).toFixed(6)})
    </p>
  `;
}

// ----------------------------
// 2) Click handler
// ----------------------------
map.on("click", (e) => {
  const lat = e.latlng.lat;
  const lon = e.latlng.lng;

  marker.setLatLng([lat, lon]);

  fetch(`${window.location.origin}/api/erosion?lat=${lat}&lon=${lon}`)
  .then(async (r) => {
    const text = await r.text();         // get raw response first
    console.log("Raw API response:", text);

    const data = JSON.parse(text);       // force parse JSON
    renderResult(data, "Map click");
  })
  .catch((err) => {
    console.error("Fetch/render error:", err);
    resultDiv.innerHTML = `
      <p style="color:#b91c1c;">
        <strong>Error:</strong> ${err}
      </p>
    `;
  });
});

// ----------------------------
// 3) ZIP handler
// ----------------------------
zipForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const z = zipInput.value.trim();
  if (!z) return;

  fetch(`${window.location.origin}/api/erosion_zip?zip=${encodeURIComponent(z)}`)
  .then(async (r) => {
    const text = await r.text();
    console.log("Raw ZIP API response:", text);

    const data = JSON.parse(text);

    if (data.error) {
      renderResult(data, `ZIP ${z}`);
      return;
    }

    map.setView([data.input_lat, data.input_lon], 11);
    marker.setLatLng([data.input_lat, data.input_lon]);
    renderResult(data, `ZIP ${data.input_zip || z}`);
  })
  .catch((err) => {
    console.error("ZIP fetch/render error:", err);
    resultDiv.innerHTML = `
      <p style="color:#b91c1c;">
        <strong>Error:</strong> ${err}
      </p>
    `;
  });
});
