# Match Cousins
## Environment
```bash
conda create -n matching python=3.10 -y
conda activate matching
pip install -r requirements.txt
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
mkdir -p deps && cd deps
git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
conda install conda-build
conda-develop . && cd ..
pip install git+https://github.com/openai/CLIP.git
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
# conda install pytorch::faiss-cpu
cd ..
```
## Usage
### sketchfab
```bash
# example
python sketchfab.py -q cup -n 10 --out cup
# download_from_sketchfab(query: str, n: int = 10, out_dir: str = "./downloads")
```
### meshyai
```bash
# example
python meshyai.py -t "spatula" -o "spatula" -n 5
python meshyai.py -i "spatula.png" -o "spatula" -n 5
# def generate_3d_model(
#     max_count: int,
#     output_dir: Path | str | None = "./meshyai_models",
#     text_prompt: Optional[str] = None,
#     image_prompt: Optional[List[Path] | List[str] | Path | str] = None,
# ) -> None:
```
### objaverse
```bash
# example
python download_objaverse.py -q spatula --out spatula -n 100
# download_from_objaverse(keyword: str, output_dir: str, max_count: int)
```

### ranking
```bash
# example
python match.py --asset_dir spatula --query_img_paths spatula.png --feature_type concat
# def rank_cousins(
#     asset_dir: str,
#     query_img_paths: str | list[str],
#     feature_type: str = "concat",
#     dinov2_backbone_size: str = "base",
#     clip_backbone_name: str = "ViT-B/16"
# )
```