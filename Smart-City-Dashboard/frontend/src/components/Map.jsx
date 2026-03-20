import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useState } from "react";
import { getBins, getFullBins } from "../services/api";
import L from "leaflet";
import axios from "axios";

// Fix marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
});

export default function Map({ bins, updateBinLevel }) {
  const [mainRoute, setMainRoute] = useState([]);
  const [truckPosition, setTruckPosition] = useState(null);
  const [animationRef, setAnimationRef] = useState(null);
  const [isCollecting, setIsCollecting] = useState(false);

  const truckStart = { lat: 20.2955, lng: 85.8240 };

  const truckIcon = new L.Icon({
    iconUrl: "https://cdn-icons-png.flaticon.com/512/743/743922.png",
    iconSize: [40, 40],
  });

  const hubIcon = new L.Icon({
    iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png", 
    iconSize: [30, 45],
    iconAnchor: [15, 45],
  });

  const getIcon = (status) => {
    let color = "green";
    if (status === "Full") color = "red";
    else if (status === "Half") color = "orange";

    return new L.Icon({
      iconUrl: `https://maps.google.com/mapfiles/ms/icons/${color}-dot.png`,
      iconSize: [32, 32],
    });
  };

  // 🌍 ORS routing
  const getRouteFromORS = async (coordinates) => {
    const apiKey = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjE0MWRiNDMyYjkwZTQyYTdhMmVmMTkwMzZhMjhmZTU0IiwiaCI6Im11cm11cjY0In0=";

    const res = await axios.post(
      "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
      { coordinates },
      {
        headers: {
          Authorization: apiKey,
          "Content-Type": "application/json",
        },
      }
    );

    return res.data.features[0].geometry.coordinates;
  };

  // 🚛 Animate truck & Reset bins
  const animateTruck = (route, fullBins) => {
    if (!route || route.length === 0) return;

    if (animationRef) clearInterval(animationRef);
    setIsCollecting(true);

    let i = 0;
    setTruckPosition(route[0]);

    // Keep track of which bins we've "collected" in this run
    let collectedBinIds = new Set();

    const interval = setInterval(() => {
      i++;

      if (i >= route.length) {
        clearInterval(interval);
        setIsCollecting(false);
        setMainRoute([]);
        return;
      }

      const currentPos = route[i];
      setTruckPosition(currentPos);

      // 🔄 Simulate collection: Check if truck is near a bin
      fullBins.forEach(bin => {
        if (collectedBinIds.has(bin.id)) return;

        const dist = Math.sqrt(Math.pow(currentPos[0] - bin.lat, 2) + Math.pow(currentPos[1] - bin.lng, 2));
        
        // Use a more generous threshold (0.005 degrees is ~500m)
        if (dist < 0.005) { 
          collectedBinIds.add(bin.id);
          updateBinLevel(bin.id, 0);
        }
      });

    }, 150); // Doubled speed for better UX

    setAnimationRef(interval);
  };

  useEffect(() => {
    // Don't restart fetching/routing if we are currently in a collection animation
    if (isCollecting) return;

    const fullBins = bins.filter(b => b.level >= 80);
    if (fullBins.length > 0) {
      const startCollection = async () => {
        setIsCollecting(true); // Set immediately to prevent race conditions
        try {
          // 🔵 CIRCULAR ROUTE: Hub -> Bins -> Hub
          const coords = [
            [truckStart.lng, truckStart.lat],
            ...fullBins.map((b) => [b.lng, b.lat]),
            [truckStart.lng, truckStart.lat], 
          ];

          const route = await getRouteFromORS(coords);
          const leafletRoute = route.map(([lng, lat]) => [lat, lng]);
          setMainRoute(leafletRoute);

          // 🚛 START MOVEMENT
          animateTruck(leafletRoute, fullBins);

        } catch (err) {
          console.error("Routing error:", err);
          setIsCollecting(false); // Reset on error so we can try again
        }
      };
      startCollection();
    } else {
      setTruckPosition([truckStart.lat, truckStart.lng]);
      setMainRoute([]);
    }
  }, [bins, isCollecting]);


  return (
    <div className="relative w-full h-[60vh] sm:h-[70vh] lg:h-full rounded-xl overflow-hidden">
      
      {/* 🧭 Legend */}
      <div className="absolute z-[1000] top-2 right-2 bg-white p-3 rounded-lg shadow-md text-xs sm:text-sm">
        <p>🚚 Route</p>
        <p>🏠 Hub</p>
        <p>🔴 Full</p>
        <p>🟠 Half</p>
        <p>🟢 Empty</p>
      </div>

      <MapContainer
        center={[20.2961, 85.8245]}
        zoom={14}
        className="w-full h-full"
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {/* 🏠 Hub Marker */}
        <Marker position={[truckStart.lat, truckStart.lng]} icon={hubIcon}>
            <Popup><b>Waste Management Hub</b><br/>Dispatching Center</Popup>
        </Marker>

        {/* 📍 Bins */}
        {bins.map((bin) => (
          <Marker
            key={bin.id}
            position={[bin.lat, bin.lng]}
            icon={getIcon(bin.status)}
          >
            <Popup>
              <div className="text-sm p-1">
                <p className="font-bold border-b mb-2">{bin.id}</p>
                <div className="flex flex-col gap-2">
                  <span>Status: <b>{bin.status}</b></span>
                  <span>Fill: <b>{bin.level}%</b></span>
                  <button
                    onClick={() => updateBinLevel(bin.id, 0)}
                    className="mt-2 w-full px-2 py-1 bg-green-600 text-white rounded text-xs font-semibold hover:bg-green-700 transition-colors"
                  >
                    Empty Now
                  </button>
                </div>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* 🚛 Moving Truck */}
        {truckPosition && (
          <Marker position={truckPosition} icon={truckIcon}>
            <Popup>🚛 Collecting Waste</Popup>
          </Marker>
        )}

        {/* 🔵 Main Route */}
        {mainRoute.length > 0 && (
          <Polyline positions={mainRoute} color="#3498db" weight={5} opacity={0.7} />
        )}
      </MapContainer>
    </div>
  );

}