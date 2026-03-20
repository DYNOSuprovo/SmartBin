"""Quick verification script for the Smart Waste Detection project."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, PATHS, CLASS_NAMES, NUM_CLASSES
print("[1/5] Config loaded OK")
print(f"  Classes: {CLASS_NAMES}")

from utils.dataset_utils import set_seed, get_class_distribution
set_seed(42)
dist = get_class_distribution(PATHS["raw_data"])
print(f"[2/5] Dataset utils OK — {sum(dist.values())} images found")

from utils.training_utils import create_model, count_parameters
m = create_model("resnet18", NUM_CLASSES, pretrained=True)
p = count_parameters(m)
print(f"[3/5] Model factory OK — ResNet18: {p['total']/1e6:.1f}M params")

from utils.inference_utils import GradCAM
print("[4/5] Inference utils OK")

from utils.simulation_utils import SmartDustbin
sb = SmartDustbin()
sb.add_waste("plastic", 0.95, "medium")
sb.add_waste("paper", 0.88, "small")
sb.add_waste("battery", 0.92, "large")
status = sb.get_status()
print("[5/5] Simulation OK")
for k, v in status.items():
    if v["item_count"] > 0:
        print(f"  {k}: {v['fill_percentage']}% ({v['item_count']} items)")

print("\n" + "="*50)
print("  ALL VERIFICATIONS PASSED!")
print("="*50)
