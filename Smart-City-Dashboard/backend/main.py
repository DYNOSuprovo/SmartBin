from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌍 GLOBAL DATA (HUB & BINS)
hub_location = {"lat": 20.2955, "lng": 85.8240}

global_bins = [
    {"id": "bin1", "lat": 20.2961, "lng": 85.8245, "level": 30},
    {"id": "bin2", "lat": 20.2975, "lng": 85.8255, "level": 50},
    {"id": "bin3", "lat": 20.2950, "lng": 85.8230, "level": 70},
    {"id": "bin4", "lat": 20.2980, "lng": 85.8265, "level": 20},
    {"id": "bin5", "lat": 20.2905, "lng": 85.8150, "level": 10},
    {"id": "bin6", "lat": 20.3050, "lng": 85.8350, "level": 15},
    {"id": "bin7", "lat": 20.2850, "lng": 85.8300, "level": 45},
    {"id": "bin8", "lat": 20.3100, "lng": 85.8100, "level": 65},
    {"id": "bin9", "lat": 20.2750, "lng": 85.8450, "level": 5},
    {"id": "bin10", "lat": 20.3200, "lng": 85.8250, "level": 40},
]


# 🏠 API: Get Hub location
@app.get("/hub")
def get_hub():
    return hub_location


# 🧠 Status logic
def update_status(bin):
    if bin["level"] >= 80:
        return "Full"
    elif bin["level"] > 40:
        return "Half"
    else:
        return "Empty"


# 📦 API: Get all bins
@app.get("/bins")
def get_bins():
    for b in global_bins:
        b["status"] = update_status(b)
    return global_bins


# 📊 API: Stats
@app.get("/stats")
def get_stats():
    return {
        "total": len(global_bins),
        "full": len([b for b in global_bins if int(b["level"]) >= 80]),
        "half": len([b for b in global_bins if 40 < int(b["level"]) < 80]),
        "empty": len([b for b in global_bins if int(b["level"]) <= 40]),
    }


# 🔔 API: Alerts
@app.get("/alerts")
def get_alerts():
    alerts = []

    for b in global_bins:
        if int(b["level"]) >= 80:
            alerts.append(f"🚨 {b['id']} is FULL")
        elif int(b["level"]) > 60:
            alerts.append(f"⚠️ {b['id']} nearing capacity")

    return alerts


# 🚛 API: Full bins (for routing)
@app.get("/full_bins")
def get_full_bins():
    return [b for b in global_bins if int(b["level"]) >= 80]



# 🎛️ API: Update bin level (IoT simulation)
@app.post("/update_bin/{bin_id}")
def update_bin(bin_id: str, level: int):
    for b in global_bins:
        if b["id"] == bin_id:
            print(f"DEBUG: Updating {bin_id} to level {level}")
            b["level"] = level
            b["status"] = update_status(b)
            return {"message": "Updated successfully", "bin": b}

    return {"error": "Bin not found"}


# 🏠 Root
@app.get("/")
def home():
    return {"message": "Smart Waste Management API Running 🚀"}