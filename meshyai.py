import requests
import os
import time
import argparse
from typing import List, Optional
from pathlib import Path
import base64
import tqdm

# MESHYAI_API = os.environ.get("MESHYAI_API")
MESHYAI_API = "msy_eN1PP1cAQApOPDN4dmVJYxnfpc0EiyHEwyss"

def generate_3d_model(
    max_count: int,
    output_dir: Path | str | None = "./meshyai_models",
    text_prompt: Optional[str] = None,
    image_prompt: Optional[List[Path] | List[str] | Path | str] = None,
) -> None:
    
    if text_prompt is None and image_prompt is None:
        raise ValueError("Either text_prompt or image_prompt must be provided.")
    if text_prompt and image_prompt:
        print("Both text_prompt and image_prompt are provided. Only text_prompt will be used.")

    if isinstance(output_dir, str):
      output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
      "Authorization": f"Bearer {MESHYAI_API}"
    }

    idx = 0

    if text_prompt:

      preview_request = {
        "mode": "preview",
        "prompt": f"one {text_prompt}",
        "negative_prompt": "low quality, low resolution, low poly, ugly",
        "art_style": "realistic",
        "should_remesh": True,
      }

      for i in tqdm.tqdm(range(max_count)):
        preview_response = requests.post(
          "https://api.meshy.ai/openapi/v2/text-to-3d",
          headers=headers,
          json=preview_request,
        )
        preview_response.raise_for_status()
        preview_task_id = preview_response.json()["result"]
        # print("Preview task created. Task ID:", preview_task_id)

        preview_task = None
        while True:
          preview_task_response = requests.get(
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{preview_task_id}",
            headers=headers,
          )
          preview_task_response.raise_for_status()
          preview_task = preview_task_response.json()
          if preview_task["status"] == "SUCCEEDED":
            # print("Preview task finished.")
            break
          # print("Preview task status:", preview_task["status"], "| Progress:", preview_task["progress"], "| Retrying in 5 seconds...")
          time.sleep(5)

        # preview_model_url = preview_task["model_urls"]["glb"]
        # preview_model_response = requests.get(preview_model_url)
        # preview_model_response.raise_for_status()

        refine_request = {
          "mode": "refine",
          "preview_task_id": preview_task_id,
          "texture_prompt": f"A highly detailed and realistic texture for a {text_prompt}",
        }
        refine_response = requests.post(
          "https://api.meshy.ai/openapi/v2/text-to-3d",
          headers=headers,
          json=refine_request,
        )
        refine_response.raise_for_status()
        refine_task_id = refine_response.json()["result"]
        # print("Refined task created. Task ID:", refined_task_id)

        refine_task = None
        while True:
          refine_task_response = requests.get(
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{refine_task_id}",
            headers=headers,
          )
          refine_task_response.raise_for_status()
          refine_task = refine_task_response.json()
          if refine_task["status"] == "SUCCEEDED":
            # print("Refined task finished.")
            break
          # print("Refined task status:", refined_task["status"], "| Progress:", refined_task["progress"], "| Retrying in 5 seconds...")
          time.sleep(5)

        # texture_url = list(refine_task["texture_urls"][0].values())[0]
        # texture_file = requests.get(texture_url)
        # texture_file.raise_for_status()
        mesh_dir = output_dir / f"{text_prompt}_{idx}"
        while mesh_dir.exists():
          idx += 1
          mesh_dir = output_dir / f"{text_prompt}_{idx}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        # with open(mesh_dir / f"{text_prompt}_{idx}.png", "wb") as f:
        #   f.write(texture_file.content)
        # mtl_response = requests.get(refine_task["model_urls"]["mtl"])
        # mtl_response.raise_for_status()
        # with open(mesh_dir / f"{text_prompt}_{idx}.mtl", "wb") as f:
        #   f.write(mtl_response.content)

        # refined_model_url = refine_task["model_urls"]["glb"]
        # refined_model_response = requests.get(refined_model_url)
        # refined_model_response.raise_for_status()
        # with open(mesh_dir / f"refined_{text_prompt}_{idx}.glb", "wb") as f:
        #   f.write(refined_model_response.content)

        remesh_request = {
          "input_task_id": refine_task_id,
          "target_formats": ["glb", "obj"],
          "resize_height": 0.01,
          "origin_at": "bottom"
        } 
        remesh_response = requests.post(
            "https://api.meshy.ai/openapi/v1/remesh",
            headers=headers,
            json=remesh_request
        )
        remesh_response.raise_for_status()
        remesh_task_id = remesh_response.json()['result']
        # print("remesh task created. Task ID:", remesh_task_id)

        remesh_task = None
        while True:
          remesh_response = requests.get(
              f"https://api.meshy.ai/openapi/v1/remesh/{remesh_task_id}",
              headers=headers,
          )
          remesh_response.raise_for_status()
          remesh_task = remesh_response.json()

          if remesh_task["status"] == "SUCCEEDED":
            # print("Remesh task finished.")
            break
          time.sleep(5)

        remeshed_glb_response = requests.get(remesh_task["model_urls"]["glb"])
        remeshed_glb_response.raise_for_status()
        with open(mesh_dir / f"{text_prompt}_{idx}.glb", "wb") as f:
          f.write(remeshed_glb_response.content)
        # remeshed_obj_response = requests.get(remesh_task["model_urls"]["obj"])
        # remeshed_obj_response.raise_for_status()
        # with open(mesh_dir / f"{text_prompt}_{idx}.obj", "wb") as f:
        #   f.write(remeshed_obj_response.content)
        idx += 1

    else:
        image_data = []
        image_exts = []
        if isinstance(image_prompt, (str, Path)):
          if isinstance(image_prompt, str):
            image_prompt = Path(image_prompt)
          with open(image_prompt, "rb") as f:
            image_data.append(base64.b64encode(f.read()).decode("utf-8"))
            image_exts.append(image_prompt.suffix.lstrip(".").lower())
        elif isinstance(image_prompt, list):
          for img in image_prompt:
              if isinstance(img, str):
                img = Path(img)
              with open(img, "rb") as f:
                image_data.append(base64.b64encode(f.read()).decode("utf-8"))
                image_exts.append(img.suffix.lstrip(".").lower())

        payload = {
            "image_urls":[
              "data:image/"+ext+";base64,"+data for ext, data in zip(image_exts, image_data)
            ],
            "should_remesh": True,
            "should_texture": True,
            "enable_pbr": True,
        }

        for i in tqdm.tqdm(range(max_count)):
          generate_response = requests.post(
              "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
              headers=headers,
              json=payload,
          )
          generate_response.raise_for_status()
          generate_task_id = generate_response.json()["result"]
          # print("3D reconstruction task created. Task ID:", task_id)

          generate_task_status = None
          while True:
            generate_task_response = requests.get(
              f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{generate_task_id}",
              headers=headers,
            )
            generate_task_response.raise_for_status()
            generate_task_status = generate_task_response.json()
            if generate_task_status["status"] == "SUCCEEDED":
              # print("Task finished.")
              break
            # print("Task status:", task_status["status"], "| Progress:", task_status["progress"], "| Retrying in 5 seconds...")
            time.sleep(5)
          
          # texture_url = list(generate_task_status["texture_urls"][0].values())[0]
          # texture_file = requests.get(texture_url)
          # texture_file.raise_for_status()
          mesh_dir = output_dir / f"reconstruction_{idx}"
          while mesh_dir.exists():
            idx += 1
            mesh_dir = output_dir / f"reconstruction_{idx}"
          mesh_dir.mkdir(parents=True, exist_ok=True)
          # with open(mesh_dir / f"reconstruction_{idx}.png", "wb") as f:
          #   f.write(texture_file.content)
          # mtl_response = requests.get(generate_task_status["model_urls"]["mtl"])
          # mtl_response.raise_for_status()
          # with open(mesh_dir / f"reconstruction_{idx}.mtl", "wb") as f:
          #   f.write(mtl_response.content)

          remesh_request = {
            "input_task_id": generate_task_id,
            "target_formats": ["glb", "obj"],
            "resize_height": 0.01,
            "origin_at": "bottom"
          } 
          remesh_response = requests.post(
              "https://api.meshy.ai/openapi/v1/remesh",
              headers=headers,
              json=remesh_request
          )
          remesh_response.raise_for_status()
          remesh_task_id = remesh_response.json()['result']
          # print("Remesh task created. Task ID:", remesh_task_id)
          remesh_task = None
          while True:
            remesh_response = requests.get(
                f"https://api.meshy.ai/openapi/v1/remesh/{remesh_task_id}",
                headers=headers,
            )
            remesh_response.raise_for_status()
            remesh_task = remesh_response.json()

            if remesh_task["status"] == "SUCCEEDED":
              # print("Remesh task finished.")
              break
            time.sleep(5)

          remeshed_glb_response = requests.get(remesh_task["model_urls"]["glb"])
          remeshed_glb_response.raise_for_status()
          with open(mesh_dir / f"reconstruction_{idx}.glb", "wb") as f:
            f.write(remeshed_glb_response.content)
          # remeshed_obj_response = requests.get(remesh_task["model_urls"]["obj"])
          # remeshed_obj_response.raise_for_status()
          # with open(mesh_dir / f"reconstruction_{idx}.obj", "wb") as f:
          #   f.write(remeshed_obj_response.content) 
          idx += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download 3D models from MeshyAI based on a keyword.")
  parser.add_argument("-t", "--text_prompt", type=str, help="Category to search for assets.")
  parser.add_argument("-o", "--output_dir", type=str, help="Directory to save generated models.")
  parser.add_argument("-n", "--max_count", type=int, help="Maximum number of models to generate.")
  parser.add_argument("-i", "--image_prompt", type=str, nargs='*', help="Path(s) to input image(s) for 3D reconstruction.")
  args = parser.parse_args()

  generate_3d_model(
    text_prompt=args.text_prompt,
    output_dir=args.output_dir,
    max_count=args.max_count,
    image_prompt=args.image_prompt,
  )