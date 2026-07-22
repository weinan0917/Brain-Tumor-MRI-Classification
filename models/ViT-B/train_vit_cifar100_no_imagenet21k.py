import sys
from pathlib import Path

from train_vit_cifar100_imagenet21k import main


def ensure_default_args():
    args = sys.argv[1:]
    if "--config" not in args:
        config_path = Path(__file__).resolve().with_name(
            "config_vit_cifar100_no_imagenet21k.yaml"
        )
        sys.argv.extend(["--config", str(config_path)])
    if "--pretrained" not in args and "--no-pretrained" not in args:
        sys.argv.append("--no-pretrained")


if __name__ == "__main__":
    ensure_default_args()
    main()
