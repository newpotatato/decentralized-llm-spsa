"""Build and audit the controlled IT routing benchmark."""
import argparse
import json
from pathlib import Path
from .it_benchmark import audit_benchmark, load_routing_benchmark, write_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-skill", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/it_routing_manifest.jsonl"))
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    records = load_routing_benchmark(args.per_skill, args.seed, args.cache_dir)
    report = audit_benchmark(records)
    write_benchmark(records, args.output, include_text=args.include_text)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()


