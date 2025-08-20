# Match Cousins
## Environment
```bash
conda create -n matching python=3.11 -y
conda activate matching
pip install -r requirements.txt
mkdir -p deps && cd deps
git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
conda install conda-build
conda-develop . && cd ..
pip install git+https://github.com/openai/CLIP.git
```
## Usage
### sketchfab
```bash
# example
python sketchfab.py -q cup -n 10 --out cup
# download_from_sketchfab(query: str, n: int = 10, out_dir: str = "./downloads")
# Log in using my account. (If you register your own account, you’ll need to complete OAuth 2.0 authentication to obtain a client ID and client secret.)
# Account: xiongjing2333@gmail.com
# Password: !1a@2s#3d$4f%5g^6h
```
### meshyai
```bash
# example
python meshyai.py -q cup -n 5 --out cup
# download_from_meshai(keyword: str, output_dir: str, max_count: int)
```
### objaverse
```bash
# example
python download_objaverse.py -q cup --out cup -n 100
# download_from_objaverse(keyword: str, output_dir: str, max_count: int)
```

### ranking
```bash
# example
python match.py --asset_dir cup --query_img_paths cup.jpeg --feature_type concat
# def rank_cousins(
#     asset_dir: str,
#     query_img_paths: str | list[str],
#     feature_type: str = "concat",
#     dinov2_backbone_size: str = "base",
#     clip_backbone_name: str = "ViT-B/16"
# )
```