#!/usr/bin/env python3
from pathlib import Path
import argparse
import objaverse
import json
from tqdm import tqdm

def find_assets(
    keyword: str,
    chunk_size: int = 5000,
    limit: int | None = None,
    match_substrings: bool = False,
) -> list[tuple[str, str, list[str], list[str]]]:
    """
    Return a list of (uid, name, categories, tags) tuples whose
    title, tags or categories match `keyword`.
    """
    keyword = keyword.lower()
    uids = objaverse.load_uids()
    hits: list[tuple[str, str, list[str], list[str]]] = []

    for start in range(0, len(uids), chunk_size):
        annots = objaverse.load_annotations(uids[start : start + chunk_size])

        for uid, a in annots.items():
            title = a.get("name", "").lower()
            cats  = [c["name"].lower() for c in a.get("categories", [])]
            tag_objs = a.get("tags", [])
            tag_names = [t["name"].lower() for t in tag_objs]
            tag_slugs = [t["slug"].lower() for t in tag_objs]
            bag = [title, *cats, *tag_names, *tag_slugs]

            def matched(field: str) -> bool:
                return keyword in field if match_substrings else field == keyword

            if any(matched(f) for f in bag):
                hits.append((uid, a.get("name", ""), cats, tag_names))
                if limit and len(hits) >= limit:
                    return hits
    return hits

def download_from_objaverse(keyword: str, output_dir: str, max_count: int) -> None: 

    assets = find_assets(keyword, 
                         limit=max_count,
                         match_substrings=True)
    
    uids = [uid for uid, _, _, _ in assets]
    local_paths = objaverse.load_objects(uids, 8)

    objaverse_download_file = f"{output_dir}/objaverse_download.jsonl"
    for uid, tmp_path in tqdm(local_paths.items(), total=len(local_paths), desc="Processing downloads"):
        path = Path(tmp_path)
        if path.suffix == ".tmp":
            finished = path.with_suffix("")
            if finished.exists():
                path = finished
            else:
                continue
        if not path.is_file():
            continue
        with open(objaverse_download_file, "a") as f:
            f.write(json.dumps({"uid": uid, "path": str(path)}) + "\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Download assets from Objaverse based on a keyword search."
  )
  parser.add_argument("-q", type=str, default="cup", help="Keyword to search for")
  parser.add_argument("--out", type=str, default="cup", help="Directory to save downloaded assets")
  parser.add_argument("-n", type=int, default=50, help="Maximum number of assets to download")
  args = parser.parse_args()

  download_from_objaverse(args.q, args.out, args.n)
