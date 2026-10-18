

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.caption_generator import CaptionGenerator


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-generate captions")
    parser.add_argument(
        "--data-root", type=str, default="./dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output", type=str, default="./data/captions/all_captions.json",
        help="Output JSON file path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root)
    gen = CaptionGenerator(seed=args.seed)

    all_captions = []
    total = 0

    for event_dir in sorted(root.iterdir()):
        if not event_dir.is_dir() or event_dir.name == "xbd_subset":
            continue
        if not (event_dir / "labels").exists():
            continue

        print(f"Processing: {event_dir.name} ...", end=" ")
        try:
            captions = gen.generate_captions_for_event(str(event_dir))
            all_captions.extend(captions)
            print(f"{len(captions)} samples")
            total += len(captions)
        except Exception as e:
            print(f"ERROR: {e}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Generated {total} captions -> {output_path}")

    # Print sample captions
    print("\n--- Sample Captions ---")
    for sample in all_captions[:3]:
        print(f"\n  Event: {sample['event']}")
        print(f"  ID:    {sample['sample_id']}")
        print(f"  Sev:   {sample['severity_score']}")
        print(f"  Caption: {sample['caption'][:200]}...")
    print("-----------------------\n")


if __name__ == "__main__":
    main()
