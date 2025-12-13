#!/usr/bin/env python3
"""
SAM-3 Inference Server

Runs in a dedicated conda environment (sam3) and serves segmentation requests
via a simple socket-based protocol.

Start with: conda run -n sam3 python models/sam3_server.py
"""

import os
import sys
import json
import base64
import socket
import struct
import pickle
import threading
from io import BytesIO
from typing import List

# Add parent directories to path for data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from PIL import Image

# Import category lists
from data.common_categories import COMMON_CATEGORIES
from data.gaming_categories import GAMING_CATEGORIES

# SAM-3 imports (only available in sam3 conda env)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_cxcywh
import torch

# Batched inference imports
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage


class SAM3Server:
    """Socket server for SAM-3 inference."""

    # Global counter for query IDs in batched inference
    _query_counter = 0

    def __init__(self, host: str = "127.0.0.1", port: int = 9999, device: str = "cuda"):
        self.host = host
        self.port = port
        self.device = device
        self.model = None
        self.processor = None
        self.socket = None
        self.running = False

        # Batched inference components (initialized in load_model)
        self.transform = None
        self.postprocessor = None

    def load_model(self):
        """Load SAM-3 model."""
        print("Loading SAM-3 model...")

        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable bfloat16 for better performance
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()

        self.model = build_sam3_image_model(
            bpe_path=None,
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
        )

        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=0.5,
        )

        # Initialize batched inference transform
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Initialize postprocessor for batched inference
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,  # No limit, filter by confidence instead
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=False,
        )

        print("SAM-3 model loaded successfully")

    def start(self):
        """Start the server."""
        self.load_model()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True

        print(f"SAM-3 server listening on {self.host}:{self.port}")

        while self.running:
            try:
                client, addr = self.socket.accept()
                print(f"Connection from {addr}")
                threading.Thread(target=self.handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def handle_client(self, client: socket.socket):
        """Handle a client connection."""
        try:
            # Read message length (4 bytes)
            length_data = client.recv(4)
            if not length_data:
                return
            msg_length = struct.unpack(">I", length_data)[0]

            # Read message
            data = b""
            while len(data) < msg_length:
                chunk = client.recv(min(msg_length - len(data), 65536))
                if not chunk:
                    break
                data += chunk

            # Parse request
            request = pickle.loads(data)

            # Process request
            response = self.process_request(request)

            # Send response
            response_data = pickle.dumps(response)
            client.sendall(struct.pack(">I", len(response_data)))
            client.sendall(response_data)

        except Exception as e:
            print(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client.close()

    def process_request(self, request: dict) -> dict:
        """Process a segmentation request."""
        try:
            method = request.get("method")

            # Handle health check first (no image required)
            if method == "health":
                return {"status": "ok", "model_loaded": self.model is not None}

            # Decode image for segmentation methods
            image_bytes = base64.b64decode(request["image_b64"])
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            if method == "segment_with_text":
                return self._segment_with_text(
                    image,
                    request["text_prompt"],
                    request.get("threshold", 0.5),
                )
            elif method == "segment_automatic":
                return self._segment_automatic(
                    image,
                    request.get("threshold", 0.5),
                )
            elif method == "segment_with_box":
                return self._segment_with_box(
                    image,
                    tuple(request["box"]),
                    request.get("threshold", 0.5),
                )
            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

        finally:
            # Always cleanup GPU memory after each request
            self._cleanup_gpu()

    @torch.no_grad()
    def _segment_with_text(self, image: Image.Image, text_prompt: str, threshold: float) -> dict:
        """Segment with text prompt."""
        self.processor.confidence_threshold = threshold
        print(f"[DEBUG] Segmenting with text prompt: '{text_prompt}', threshold: {threshold}")

        state = self.processor.set_image(image)
        print(f"[DEBUG] Image size: {image.size}")
        state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

        result = self._extract_results(state, label=text_prompt)

        # Clear processor state to free GPU memory
        del state
        return result

    @torch.no_grad()
    def _segment_automatic(self, image: Image.Image, threshold: float) -> dict:
        """Automatic segmentation using common + gaming categories.

        TRUE BATCHED: All prompts processed in a SINGLE forward pass.
        """
        import time
        start_time = time.time()

        # Combine all categories
        all_categories = COMMON_CATEGORIES + GAMING_CATEGORIES

        # Build UNIQUE prompt list (deduplicate names and categories)
        prompts = []
        seen = set()

        for cat in all_categories:
            name = cat["name"]
            if name not in seen:
                prompts.append(name)
                seen.add(name)

            category = cat["category"]
            if category not in seen:
                prompts.append(category)
                seen.add(category)

        print(f"[BATCHED] Running automatic segmentation with {len(prompts)} unique prompts")

        # Update postprocessor threshold
        self.postprocessor.detection_threshold = threshold

        # Create datapoint with image and ALL prompts
        w, h = image.size
        datapoint = Datapoint(find_queries=[], images=[])
        datapoint.images = [SAMImage(data=image, objects=[], size=[h, w])]

        # Track query IDs for each prompt
        prompt_to_query_id = {}
        for prompt in prompts:
            SAM3Server._query_counter += 1
            query_id = SAM3Server._query_counter

            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query_id,
                        original_image_id=query_id,
                        original_category_id=1,
                        original_size=[h, w],  # [height, width] format expected by postprocessor
                        object_id=0,
                        frame_index=0,
                    )
                )
            )
            prompt_to_query_id[prompt] = query_id

        # Transform the datapoint
        datapoint = self.transform(datapoint)

        # Collate into batch and move to GPU
        batch = collate([datapoint], dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        print(f"[BATCHED] Datapoint prepared in {time.time() - start_time:.2f}s")
        forward_start = time.time()

        # SINGLE forward pass for ALL prompts!
        output = self.model(batch)

        print(f"[BATCHED] Forward pass completed in {time.time() - forward_start:.2f}s")

        # Post-process results
        processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

        # Convert results to our segment format
        all_segments = []
        for prompt, query_id in prompt_to_query_id.items():
            if query_id not in processed_results:
                continue

            result = processed_results[query_id]
            segments = self._convert_postprocessed_result(result, prompt, (w, h))
            if segments:
                all_segments.extend(segments)
                print(f"[BATCHED] Prompt '{prompt}' found {len(segments)} segments")

        # Explicitly delete large tensors to free GPU memory
        del batch
        del output
        del processed_results
        del datapoint

        # Remove duplicates based on IoU
        all_segments = self._remove_duplicates(all_segments)
        print(f"[BATCHED] Total after dedup: {len(all_segments)} segments")
        print(f"[BATCHED] Total time: {time.time() - start_time:.2f}s")

        return {"segments": all_segments}

    def _convert_postprocessed_result(
        self,
        result: dict,
        label: str,
        img_size: tuple,
    ) -> List[dict]:
        """Convert postprocessed result to our segment format."""
        segments = []

        masks = result.get("masks")
        boxes = result.get("boxes")
        scores = result.get("scores")

        if masks is None or boxes is None or scores is None:
            return segments

        # Handle tensor vs list
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        for i in range(len(masks)):
            mask = masks[i]
            if len(mask.shape) == 3:
                mask = mask[0]  # Remove channel dim if present
            mask = mask.astype(np.uint8)

            box = boxes[i]
            score = float(scores[i])

            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            # Encode mask as base64
            mask_bytes = mask.tobytes()

            segments.append({
                "mask_b64": base64.b64encode(mask_bytes).decode(),
                "mask_shape": mask.shape,
                "bbox": bbox,
                "score": score,
                "label": label,
            })

        return segments

    @torch.no_grad()
    def _segment_with_box(self, image: Image.Image, box: tuple, threshold: float) -> dict:
        """Segment with box prompt."""
        self.processor.confidence_threshold = threshold

        state = self.processor.set_image(image)
        self.processor.reset_all_prompts(state)

        # Convert box from xyxy to normalized cxcywh
        img_w, img_h = image.size
        x1, y1, x2, y2 = box
        box_xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        box_cxcywh = box_xyxy_to_cxcywh(box_xyxy)

        # Normalize to [0, 1]
        norm_box = box_cxcywh[0].tolist()
        norm_box[0] /= img_w
        norm_box[1] /= img_h
        norm_box[2] /= img_w
        norm_box[3] /= img_h

        state = self.processor.add_geometric_prompt(
            state=state,
            box=norm_box,
            label=True,
        )

        result = self._extract_results(state)

        # Clear processor state to free GPU memory
        del state
        del box_xyxy
        del box_cxcywh
        return result

    def _extract_results(self, state: dict, label: str = "object") -> dict:
        """Extract results from processor state."""
        segments = []

        # Debug: print state keys
        print(f"[DEBUG] State keys: {state.keys()}")

        if "masks" not in state:
            print("[DEBUG] No 'masks' key in state")
            return {"segments": segments}

        if state["masks"] is None or len(state["masks"]) == 0:
            print("[DEBUG] masks is None or empty")
            return {"segments": segments}

        masks = state["masks"].cpu().numpy()
        boxes = state["boxes"].cpu().numpy()
        scores = state["scores"].cpu().numpy()

        print(f"[DEBUG] Found {len(masks)} masks")
        print(f"[DEBUG] Original image size: {state.get('original_width')}x{state.get('original_height')}")

        for i in range(len(masks)):
            mask = masks[i, 0]  # Remove channel dim
            box = boxes[i]
            score = float(scores[i])

            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            print(f"[DEBUG] Object {i}: bbox={bbox}, score={score:.2f}, label={label}, mask_shape={mask.shape}")

            # Encode mask as base64 for transmission
            mask_bytes = mask.astype(np.uint8).tobytes()

            segments.append({
                "mask_b64": base64.b64encode(mask_bytes).decode(),
                "mask_shape": mask.shape,
                "bbox": bbox,
                "score": score,
                "label": label,
            })

        return {"segments": segments}

    def _remove_duplicates(self, segments: list, iou_threshold: float = 0.7) -> list:
        """Remove duplicate segments."""
        if len(segments) <= 1:
            return segments

        # Sort by score
        segments = sorted(segments, key=lambda x: x["score"], reverse=True)

        keep = []
        for seg in segments:
            is_dup = False
            for kept in keep:
                # Simple bbox IoU check
                iou = self._bbox_iou(seg["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(seg)

        return keep

    def _bbox_iou(self, box1: list, box2: list) -> float:
        """Compute bbox IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _cleanup_gpu(self):
        """
        Clean up GPU memory after each request.

        Clears cached tensors and forces garbage collection to prevent
        memory accumulation across requests.
        """
        import gc

        # Clear processor state if it has any cached data
        if self.processor is not None:
            # Reset any cached image encodings
            if hasattr(self.processor, 'state') and self.processor.state is not None:
                self.processor.state = None
            if hasattr(self.processor, '_cached_image'):
                self.processor._cached_image = None

        # Force garbage collection
        gc.collect()
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM-3 Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind to")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    server = SAM3Server(host=args.host, port=args.port, device=args.device)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()
