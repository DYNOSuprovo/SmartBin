export default function Dashboard({ bins }) {
  const stats = {
    total: bins.length,
    full: bins.filter((b) => b.level >= 80).length,
    half: bins.filter((b) => b.level > 40 && b.level < 80).length,
    empty: bins.filter((b) => b.level <= 40).length,
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total */}
      <div className="bg-white p-4 rounded-xl shadow flex flex-col justify-between">
        <p className="text-gray-500 text-sm sm:text-base">Total Bins</p>
        <h2 className="text-xl sm:text-2xl font-bold">{stats.total}</h2>
      </div>

      {/* Full */}
      <div className="bg-red-500 text-white p-4 rounded-xl shadow flex flex-col justify-between">
        <p className="text-sm sm:text-base">Full</p>
        <h2 className="text-xl sm:text-2xl font-bold">{stats.full}</h2>
      </div>

      {/* Half */}
      <div className="bg-yellow-400 p-4 rounded-xl shadow flex flex-col justify-between">
        <p className="text-sm sm:text-base">Half</p>
        <h2 className="text-xl sm:text-2xl font-bold">{stats.half}</h2>
      </div>

      {/* Empty */}
      <div className="bg-green-500 text-white p-4 rounded-xl shadow flex flex-col justify-between">
        <p className="text-sm sm:text-base">Empty</p>
        <h2 className="text-xl sm:text-2xl font-bold">{stats.empty}</h2>
      </div>
    </div>
  );
}