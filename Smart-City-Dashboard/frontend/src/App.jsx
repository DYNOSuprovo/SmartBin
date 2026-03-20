import { useState, useEffect } from "react";
import Map from "./components/Map";
import Dashboard from "./components/Dashboard";
import Alerts from "./components/Alerts";
import BinControl from "./components/BinControl";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { getBins } from "./services/api";

function App() {
  const [bins, setBins] = useState([]);

  const fetchBins = async () => {
    const data = await getBins();
    setBins(data);
  };

  const updateBinLevel = (id, level) => {
    // Immediate UI update
    setBins((prev) =>
      prev.map((b) =>
        b.id === id ? { ...b, level: parseInt(level), status: level >= 80 ? "Full" : level > 40 ? "Half" : "Empty" } : b
      )
    );
    console.log(`DEBUG: Requesting update for ${id} to ${level}%`);
    // API Call
    fetch(`http://127.0.0.1:8000/update_bin/${id}?level=${level}`, { method: "POST" })
      .then(res => res.json())
      .then(data => console.log("DEBUG: Update response:", data))
      .catch(err => console.error("DEBUG: Update error:", err));
  };

  useEffect(() => {
    fetchBins();
    const interval = setInterval(fetchBins, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-gray-100 min-h-screen p-4 sm:p-6">
      <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-4 sm:mb-6">
        🚀 Smart Waste Management Dashboard
      </h1>

      {/* Dashboard Stats */}
      <Dashboard bins={bins} />

      {/* Hidden Alerts Logic (Toast only) */}
      <Alerts bins={bins} />

      <ToastContainer position="top-right" autoClose={3000} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-3 sm:p-4 flex flex-col">
          <h2 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">🗺️ Live Map</h2>
          <div className="w-full flex-1 min-h-[300px] sm:min-h-[400px] lg:min-h-[500px]">
            <Map bins={bins} updateBinLevel={updateBinLevel} />
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-4">
          <h2 className="text-lg sm:text-xl font-semibold mb-3">🎛️ Bin Control</h2>
          <BinControl bins={bins} updateBinLevel={updateBinLevel} />
        </div>
      </div>
    </div>
  );
}


export default App;