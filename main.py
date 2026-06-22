
import sys
from collect import collect_and_identify
from train import train_sac, test_visual

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Commands: collect, train_with, train_without, test_with, test_without")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "collect":
        collect_and_identify()
    elif cmd == "train_with":
        train_sac(use_drem=True, num_steps=300_000)
    elif cmd == "train_without":
        train_sac(use_drem=False, num_steps=300_000)
    elif cmd == "test_with":
        test_visual("with_drem")
    elif cmd == "test_without":
        test_visual("without_drem")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
