"""Prepare a balanced IT-specialized routing dataset."""
import argparse
from pathlib import Path
from .it_dataset_loader import ITSkill, load_it_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-skill", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/it_tasks.jsonl"))
    parser.add_argument("--skills", default=",".join(skill.value for skill in ITSkill))
    args = parser.parse_args()
    skills = [ITSkill(x.strip()) for x in args.skills.split(",") if x.strip()]
    records = load_it_dataset(args.per_skill, args.seed, skills, args.cache_dir)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
