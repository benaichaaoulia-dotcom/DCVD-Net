"""Allow ``python -m dcvd_net`` to show package command help."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DCVD-Net utilities are exposed as dcvd-train, dcvd-evaluate, "
            "and dcvd-predict."
        )
    )
    parser.print_help()


if __name__ == "__main__":
    main()
