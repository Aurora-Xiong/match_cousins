#!/usr/bin/env python3
"""
Rank all images in a directory by feature similarity to a query image,
"""
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
    """Load an image as a uint8 H×W×3 array."""
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

    # ---------- DINOv2 ----------
    if dino_encoder is not None:
        # (1,C,H,W) → patch-feats → (Hf,Wf,D)
        dino_feats = dino_encoder.get_features(img_np)          
        dino_feats = dino_feats.squeeze(0)                       # → (Hf, Wf, D)
        Hf, Wf, D = dino_feats.shape
        dino_feats = dino_feats.reshape(-1, D)              # → (Hf*Wf, D)
        dino_feats = dino_feats.mean(axis=0)                 # → (D,)
        desc_list.append(dino_feats)

    # ---------- CLIP ----------
    if clip_encoder is not None:
        x = clip_encoder.preprocess(img_np).to(clip_encoder.device)   # (1,C,H,W)
        clip_feats = clip_encoder.forward(x).squeeze(0)               # (D,)
        clip_feats = clip_feats.detach().cpu().numpy()                # → ndarray
        desc_list.append(clip_feats)

    if not desc_list:
        raise ValueError("Both encoders are None.")

    desc = np.concatenate(desc_list, axis=0) if len(desc_list) > 1 else desc_list[0]

    return np.ascontiguousarray(desc, dtype=np.float32)

def main():
    p = argparse.ArgumentParser(description="Rank images by DINOv2 feature similarity (digital-cousins)")
    p.add_argument("-q", "--query", type=Path, help="Path to the query object image")
    p.add_argument("-c", "--candidates", type=Path, help="Directory containing candidate images")
    p.add_argument("--feature_type", default="concat", choices=["dino_v2", "clip", "concat"], help="Feature type to use for encoding images")
    p.add_argument("--dinov2_backbone_size", default="base", choices=DinoV2Encoder.BACKBONE_ARCHES.keys(), help="One of small/base/large/giant")
    p.add_argument("--clip_backbone_name", default="ViT-B/16", choices=CLIPEncoder.EMBEDDING_DIMS.keys(), help="One of ViT-B/16, ViT-B/32, ViT-L/14, etc.")
    p.add_argument("--device", default=None, help="torch device (e.g. 'cpu' or 'cuda'); defaults to cuda if available")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Initialize the encoder
    print(f"[INFO] Loading Encoder on {device}")
    if args.feature_type == "dino_v2":
        dino_encoder = DinoV2Encoder(
            backbone_size=args.dinov2_backbone_size,
            device=device
        )
        clip_encoder = None
        dino_encoder.eval()
    elif args.feature_type == "clip":
        clip_encoder = CLIPEncoder(
            backbone_name=args.clip_backbone_name,
            device=device
        )
        dino_encoder = None
        clip_encoder.eval()
    elif args.feature_type == "concat":
        dino_encoder = DinoV2Encoder(
            backbone_size=args.dinov2_backbone_size,
            device=device
        )
        clip_encoder = CLIPEncoder(
            backbone_name=args.clip_backbone_name,
            device=device
        )
        dino_encoder.eval()
        clip_encoder.eval()

    # 2) Compute descriptor for query
    print(f"[INFO] Encoding query image: {args.query}")
    query_img = load_rgb_image(args.query)
    query_vec = compute_global_descriptor(dino_encoder, clip_encoder, query_img)

    # 3) Encode all candidates
    print(f"[INFO] Encoding candidates under: {args.candidates}")
    cand_vecs  = []       
    cand_paths = []

    img_dir = Path(args.candidates)
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    for img_file in img_files:
        # print(f"[INFO] Encoding {img_file}")
        img = load_rgb_image(img_file)
        vec = compute_global_descriptor(dino_encoder, clip_encoder, img)
        cand_vecs.append(vec)
        cand_paths.append(img_file)

    cand_vecs = np.stack(cand_vecs, axis=0)       # 形状 (N, D)

    # 4) Search & rank
    D = query_vec.shape[0]
    index = faiss.IndexFlatL2(D)
    index.add(cand_vecs)

    Dists, Idxs = index.search(query_vec[None, :], len(cand_paths))
    Dists, Idxs = Dists[0], Idxs[0]

    # 5) print results
    print("\nRanked candidates (most similar first):")
    for rank, (idx, dist) in enumerate(zip(Idxs, Dists), start=1):
        p = cand_paths[idx]
        print(f"{rank:2d}. {p.parent.name}/{p.name:<25s}  dist = {dist:.4f}")

    out_path = Path("match_rankings") / f"{args.query.stem}_ranking.txt"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        f.write("Ranked candidates (most similar first):\n")
        for rank, (idx, dist) in enumerate(zip(Idxs, Dists), start=1):
            p = cand_paths[idx]
            f.write(f"{rank:2d}. {p.parent.name}/{p.name:<25s}  dist = {dist:.4f}\n")
    print(f"\n[INFO] Saved ranking results to: {out_path}")

if __name__ == "__main__":
    main()


