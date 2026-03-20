export default function BinControl({ bins, updateBinLevel }) {
  return (
    <div className="bg-white p-4 rounded-xl shadow mt-6">
      <h2 className="text-lg font-semibold mb-3">🎛️ IoT Bin Control</h2>

      {bins.map((bin) => (
        <div key={bin.id} className="mb-4">
          <p className="text-sm font-medium">{bin.id} ({bin.level}%)</p>

          <div className="flex items-center gap-3">
            <input
              type="range"
              min="0"
              max="100"
              value={bin.level || 0}
              onChange={(e) => updateBinLevel(bin.id, e.target.value)}
              className="flex-1 transition-all cursor-pointer"
            />
            <button
              onClick={() => updateBinLevel(bin.id, 0)}
              className="px-2 py-1 bg-red-100 text-red-600 text-xs font-bold rounded hover:bg-red-200 transition-colors"
            >
              Empty
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}