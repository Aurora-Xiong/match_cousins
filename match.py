#!/usr/bin/env python3
import argparse
from pathlib import Path
import faiss
from PIL import Image
import os
import bpy
import numpy as np
import torch
from models.dino_v2 import DinoV2Encoder
from models.clip import CLIPEncoder
from mathutils import Vector
from tqdm import tqdm
import math
import json

def render_blender(glb_path, out_dir, size=512):
    """
    Render six-view PNGs (+X, -X, +Y, -Y, +Z, -Z) of a GLB model.
    Each image uses white background, transparency, and strong lighting.
    """
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Load .glb model
    bpy.ops.import_scene.gltf(filepath=glb_path)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    obj = bpy.context.active_object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

    # Compute camera distance
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    corners = [m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    bbox_min = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    bbox_max = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (bbox_min + bbox_max) * 0.5
    extent = (bbox_max - bbox_min).length
    radius = extent * 1.2

    # Add camera
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(45)

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.cycles.device = 'GPU' if bpy.context.preferences.addons['cycles'].preferences.compute_device_type != 'NONE' else 'CPU'

    # White background
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.light_settings.ao_factor = 0.6
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.92, 0.92, 0.92, 1)
    bg.inputs[1].default_value = 1.0

    # Add 3 bright directional lights
    light_dirs = [(1, 1, 1), (-1, 1, 1), (0, -1, 1)]
    for i, dir_vec in enumerate(light_dirs):
        light_data = bpy.data.lights.new(name=f'Light{i}', type='SUN')
        light_data.energy = 1.0
        light = bpy.data.objects.new(f'Light{i}', light_data)
        bpy.context.collection.objects.link(light)
        light.rotation_mode = 'QUATERNION'
        light.location = (0, 0, 0)
        light.rotation_euler = (
            math.atan2(dir_vec[1], dir_vec[2]),
            -math.atan2(dir_vec[0], dir_vec[2]),
            0
        )

    # Define six view directions
    views = {
        "posx": Vector((1, 0, 0)),
        "negx": Vector((-1, 0, 0)),
        "posy": Vector((0, 1, 0)),
        "negy": Vector((0, -1, 0)),
        "posz": Vector((0, 0, 1)),
        "negz": Vector((0, 0, -1)),
    }

    def look_at(cam, target):
        quat = (target - cam.location).to_track_quat('-Z', 'Y')
        cam.rotation_euler = quat.to_euler()

    # Render each view
    for name, direction in views.items():
        cam.location = center + direction * radius
        look_at(cam, center)
        scene.render.filepath = os.path.join(out_dir, f"{name}.png")
        bpy.ops.render.render(write_still=True)

def load_rgb_image(fpath: Path) -> np.ndarray:
    img = Image.open(fpath).convert("RGB")
    return np.array(img, dtype=np.uint8)

def compute_global_descriptor(
    dino_encoder: DinoV2Encoder,
    clip_encoder: CLIPEncoder,
    img_np: np.ndarray
) -> np.ndarray:
    """
    Encode the image into patch features and mean-pool into a D-dim vector.
    """
    desc_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- DINOv2 ----------
    if dino_encoder is not None:
        # (1,C,H,W) → patch-feats → (Hf,Wf,D)
        dino_feats = dino_encoder.get_features(img_np)          
        dino_feats = dino_feats.squeeze(0)                       # → (Hf, Wf, D)
        Hf, Wf, D = dino_feats.shape
        dino_feats = dino_feats.reshape(-1, D)              # → (Hf*Wf, D)
        dino_feats = dino_feats.mean(axis=0)                 # → (D,)
        desc_list.append(torch.tensor(dino_feats, device=device))

    # ---------- CLIP ----------
    if clip_encoder is not None:
        x = clip_encoder.preprocess(img_np).to(clip_encoder.device)   # (1,C,H,W)
        clip_feats = clip_encoder.forward(x).squeeze(0)               # (D,)
        desc_list.append(clip_feats)

    if not desc_list:
        raise ValueError("Both encoders are None.")

    if len(desc_list) == 1:
        desc = desc_list[0]
    else:
        desc = torch.cat(desc_list, dim=0)   # (D1 + D2,)

    return desc.detach().cpu().numpy().astype(np.float32)

def rank_cousins(
    asset_dir: str,
    query_img_paths: str | list[str],
    feature_type: str = "concat",
    dinov2_backbone_size: str = "base",
    clip_backbone_name: str = "ViT-B/16"
):
    assert feature_type in ["dino_v2", "clip", "concat"], f"Invalid feature type: {feature_type}"

    # 1.render images of cousins
    files = os.listdir(asset_dir)
    image_dir = f"{asset_dir}_images"
    os.makedirs(image_dir, exist_ok=True)
    for file in tqdm(files, total=len(files), desc="Rendering candidates"):
        if file.endswith(".glb"):
            glb_path = os.path.join(asset_dir, file)
            render_blender(glb_path, f"{image_dir}/{file.replace('.glb', '')}", size=512)

    objaverse_data = {}
    with open(f"{asset_dir}/objaverse_download.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            objaverse_data[data["uid"]] = data["path"]
    
    for uid in objaverse_data.keys():
        glb_path = objaverse_data[uid]
        render_blender(glb_path, f"{image_dir}/{uid}", size=512)

    # 2.rank images
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Initialize the encoder
    if feature_type == "dino_v2":
        dino_encoder = DinoV2Encoder(
            backbone_size=dinov2_backbone_size,
            device=device
        )
        dino_encoder.eval()
        clip_encoder = None
    elif feature_type == "clip":
        clip_encoder = CLIPEncoder(
            backbone_name=clip_backbone_name,
            device=device
        )
        dino_encoder = None
        clip_encoder.eval()
    elif feature_type == "concat":
        dino_encoder = DinoV2Encoder(
            backbone_size=dinov2_backbone_size,
            device=device
        )
        clip_encoder = CLIPEncoder(
            backbone_name=clip_backbone_name,
            device=device
        )
        dino_encoder.eval()
        clip_encoder.eval()

    if isinstance(query_img_paths, str):
        query_img_paths = [query_img_paths]

    # 2) Encode all candidates
    cand_vecs = []
    cand_paths = []
    cand_image_paths = []
    dir_lists = os.listdir(image_dir)
    for dir_name in tqdm(dir_lists, total=len(dir_lists), desc="Encoding candidates"):
        dir_path = os.path.join(image_dir, dir_name)
        img_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for img_file in img_files:
            img = load_rgb_image(os.path.join(dir_path, img_file))
            vec = compute_global_descriptor(dino_encoder, clip_encoder, img)
            cand_vecs.append(vec)
            cand_image_paths.append(os.path.join(dir_path, img_file))
            if os.path.exists(os.path.join(asset_dir, f"{dir_name}.glb")):
                cand_paths.append(os.path.join(asset_dir, f"{dir_name}.glb"))
            else:
                cand_paths.append(objaverse_data[dir_name])
    cand_vecs = np.stack(cand_vecs, axis=0).astype(np.float32)       # 形状 (N, D)

    # 3) Encode query images
    query_vecs = []
    for qpath in query_img_paths:
        print(f"[INFO] Encoding query image: {qpath}")
        qimg = load_rgb_image(qpath)
        qvec = compute_global_descriptor(dino_encoder, clip_encoder, qimg)  # -> (D,)
        query_vecs.append(qvec)
    query_vecs = np.stack(query_vecs, axis=0).astype(np.float32)

    # 4) Search & rank
    D = cand_vecs.shape[1]
    index = faiss.IndexFlatL2(D)
    index.add(cand_vecs)

    V = cand_vecs.shape[0]
    best_dists = np.full(V, np.inf, dtype=np.float32)

    for qi in range(query_vecs.shape[0]):
        Dists, Idxs = index.search(query_vecs[qi:qi+1], V)  # (1, V)
        drow, irow = Dists[0], Idxs[0]
        mask = drow < best_dists[irow]
        best_dists[irow[mask]] = drow[mask]

    order = np.argsort(best_dists)
    sorted_cand_paths = [cand_paths[i] for i in order]
    sorted_cand_image_paths = [cand_image_paths[i] for i in order]
    sorted_best_dists = best_dists[order]

    with open(f"{asset_dir}/ranking.jsonl", "w") as f:
        for i, (cand_path, cand_image_path) in enumerate(zip(sorted_cand_paths, sorted_cand_image_paths)):
            f.write(json.dumps({
                "rank": i + 1,
                "glb_path": cand_path,
                "image_path": cand_image_path,
                "distance": float(sorted_best_dists[i])
            }) + "\n")
        
def main():
    parser = argparse.ArgumentParser(description="Rank cousins based on image similarity.")
    parser.add_argument("--asset_dir", type=str, required=True, help="Directory containing .glb files.")
    parser.add_argument("--query_img_paths", type=str, nargs='+', required=True, help="Path(s) to query image(s).")
    parser.add_argument("--feature_type", type=str, default="concat", choices=["dino_v2", "clip", "concat"], help="Feature type to use for encoding.")
    parser.add_argument("--dinov2_backbone_size", type=str, default="base", help="DINOv2 backbone size.")
    parser.add_argument("--clip_backbone_name", type=str, default="ViT-B/16", help="CLIP backbone name.")

    args = parser.parse_args()
    
    rank_cousins(
        asset_dir=args.asset_dir,
        query_img_paths=args.query_img_paths,
        feature_type=args.feature_type,
        dinov2_backbone_size=args.dinov2_backbone_size,
        clip_backbone_name=args.clip_backbone_name
    )

if __name__ == "__main__":
    main()


