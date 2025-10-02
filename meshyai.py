import requests
import os
import time
import argparse
from typing import List, Optional
from pathlib import Path
import base64
import tqdm

# MESHYAI_API = os.environ.get("MESHYAI_API")
MESHYAI_API = "msy_6Y8nCVSACVEDwuQlsikIG71WzP3TRkeNDCxY"

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

      generate_preview_request = {
        "mode": "preview",
        "prompt": f"one {text_prompt}",
        "negative_prompt": "low quality, low resolution, low poly, ugly",
        "art_style": "realistic",
        "should_remesh": True,
      }

      for i in tqdm.tqdm(range(max_count)):

        # 1. Generate a preview model and get the task ID
        generate_preview_response = requests.post(
          "https://api.meshy.ai/openapi/v2/text-to-3d",
          headers=headers,
          json=generate_preview_request,
        )
        generate_preview_response.raise_for_status()
        preview_task_id = generate_preview_response.json()["result"]
        # print("Preview task created. Task ID:", preview_task_id)

        # 2. Poll the preview task status until it's finished
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

        # 3. Download the preview model in glb format
        preview_model_url = preview_task["model_urls"]["glb"]
        preview_model_response = requests.get(preview_model_url)
        preview_model_response.raise_for_status()

        # with open(f"{output_dir}/meshyai_preview_model_{i}.glb", "wb") as f:
        #   f.write(preview_model_response.content)

        # 4. Generate a refined model and get the task ID
        generate_refined_request = {
          "mode": "refine",
          "preview_task_id": preview_task_id,
        }
        generate_refined_response = requests.post(
          "https://api.meshy.ai/openapi/v2/text-to-3d",
          headers=headers,
          json=generate_refined_request,
        )
        generate_refined_response.raise_for_status()
        refined_task_id = generate_refined_response.json()["result"]
        # print("Refined task created. Task ID:", refined_task_id)

        # 5. Poll the refined task status until it's finished
        refined_task = None
        while True:
          refined_task_response = requests.get(
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{refined_task_id}",
            headers=headers,
          )
          refined_task_response.raise_for_status()
          refined_task = refined_task_response.json()
          if refined_task["status"] == "SUCCEEDED":
            # print("Refined task finished.")
            break
          # print("Refined task status:", refined_task["status"], "| Progress:", refined_task["progress"], "| Retrying in 5 seconds...")
          time.sleep(5)

        # 6. Download the refined model in glb format
        refined_model_url = refined_task["model_urls"]["glb"]
        remesh_request = {
          "model_url": refined_model_url,
          "target_formats": ["glb"],
          "resize_height": 0.01,
          "origin_at": "bottom"
        } 
        generate_remesh_response = requests.post(
            "https://api.meshy.ai/openapi/v1/remesh",
            headers=headers,
            json=remesh_request
        )
        generate_remesh_response.raise_for_status()
        remesh_task_id = generate_remesh_response.json()['result']
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

        remeshed_model_response = requests.get(remesh_task["model_urls"]["glb"])
        remeshed_model_response.raise_for_status()

        output_path = output_dir / f"{text_prompt}_{idx}.glb"
        while output_path.exists():
          idx += 1
          output_path = output_dir / f"{text_prompt}_{idx}.glb"
        with open(output_path, "wb") as f:
          f.write(remeshed_model_response.content)
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
          response = requests.post(
              "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
              headers=headers,
              json=payload,
          )
          response.raise_for_status()
          task_id = response.json()["result"]
          # print("3D reconstruction task created. Task ID:", task_id)

          task_status = None
          while True:
            task_response = requests.get(
              f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{task_id}",
              headers=headers,
            )
            task_response.raise_for_status()
            task_status = task_response.json()
            if task_status["status"] == "SUCCEEDED":
              # print("Task finished.")
              break
            # print("Task status:", task_status["status"], "| Progress:", task_status["progress"], "| Retrying in 5 seconds...")
            time.sleep(5)
          
          model_url = task_status["model_urls"]["glb"]
          remesh_request = {
            "model_url": model_url,
            "target_formats": ["glb"],
            "resize_height": 0.01,
            "origin_at": "bottom"
          } 
          generate_remesh_response = requests.post(
              "https://api.meshy.ai/openapi/v1/remesh",
              headers=headers,
              json=remesh_request
          )
          generate_remesh_response.raise_for_status()
          remesh_task_id = generate_remesh_response.json()['result']
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

          model_response = requests.get(remesh_task["model_urls"]["glb"])
          model_response.raise_for_status()
          output_path = output_dir / f"reconstruction_{i}.glb"
          while output_path.exists():
            idx += 1
            output_path = output_dir / f"reconstruction_{idx}.glb"
          with open(output_path, "wb") as f:
            f.write(model_response.content)
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