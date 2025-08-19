# Match Cousins
## Environment
```bash
conda create -n matching python=3.11 -y
conda activate matching
pip install -r requirements.txt
mkdir -p deps && cd deps
git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
conda-develop . && cd ..
pip install git+https://github.com/openai/CLIP.git
```
## Usage
### sketchfab
```bash
# example
python sketchfab.py -q cup -n 10 --out cup
# download_from_sketchfab(query: str, n: int = 10, out_dir: str = "./downloads")
```
## meshyai
```bash
# example
python meshyai.py -q cup -n 10 --out cup
# download_from_meshai(keyword: str, output_dir: str, max_count: int)
```
## objaverse
```bash
# example
python download_objaverse.py -q cup --out cup -n 100
# download_from_objaverse(keyword: str, output_dir: str, max_count: int)
```