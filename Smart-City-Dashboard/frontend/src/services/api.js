export const getBins = async () => {
  try {
    const res = await fetch("http://127.0.0.1:8000/bins");
    return await res.json();
  } catch (err) {
    console.error("Error fetching bins:", err);
    return [];
  }
};

export const getFullBins = async () => {
  try {
    const res = await fetch("http://127.0.0.1:8000/full_bins");
    return await res.json();
  } catch (err) {
    console.error(err);
    return [];
  }
};