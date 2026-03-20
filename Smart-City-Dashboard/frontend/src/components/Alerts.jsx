import { useEffect, useRef } from "react";
import { toast } from "react-toastify";


export default function Alerts({ bins }) {
  const notifiedBins = useRef(new Set());

  useEffect(() => {
    bins.forEach((bin) => {
      const binKey = `${bin.id}-${bin.level}`;
      
      if (bin.level >= 80 && !notifiedBins.current.has(bin.id)) {
        toast.error(`⚠️ ${bin.id} is FULL! (${bin.level}%)`);
        notifiedBins.current.add(bin.id);
      } 
      
      // Reset notification flag when bin is emptied
      if (bin.level < 40 && notifiedBins.current.has(bin.id)) {
        notifiedBins.current.delete(bin.id);
      }
    });
  }, [bins]);

  return null;
}