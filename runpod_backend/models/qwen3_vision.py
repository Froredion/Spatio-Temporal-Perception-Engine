"""
Qwen3-VL Model Wrapper

Vision Language Model providing:
- Multi-image reasoning
- Caption generation
- Query answering
- Video frame sequence analysis
- Parallel batch inference with async processing

Key specs:
- 8B parameters
- Up to 128K token context
- Native multi-image support
- Works with transformers>=4.56.0
"""

import torch
from typing import Dict, List, Optional, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class Qwen3VisionModel:
    """Qwen3-VL inference wrapper for vision-language tasks."""

    # Max resolution for processing (smaller = faster)
    MAX_IMAGE_SIZE = 512

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to max size while preserving aspect ratio."""
        w, h = image.size
        if max(w, h) <= self.MAX_IMAGE_SIZE:
            return image

        if w > h:
            new_w = self.MAX_IMAGE_SIZE
            new_h = int(h * self.MAX_IMAGE_SIZE / w)
        else:
            new_h = self.MAX_IMAGE_SIZE
            new_w = int(w * self.MAX_IMAGE_SIZE / h)

        return image.resize((new_w, new_h), Image.LANCZOS)

    def load(self):
        """Load Qwen3-VL model and processor."""
        if self._loaded:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"Loading Qwen3-VL from {self.config.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        self._loaded = True
        print("Qwen3-VL loaded successfully")

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            torch.cuda.empty_cache()

    def _build_messages(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> List[Dict]:
        """Build chat messages for Qwen3-VL."""
        content = []

        # Add images
        for img in images:
            content.append({
                "type": "image",
                "image": img,
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt,
        })

        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def generate(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text response for image(s) and prompt.

        Args:
            images: Single PIL Image or list of PIL Images
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated text response
        """
        if not self._loaded:
            self.load()

        if isinstance(images, Image.Image):
            images = [images]

        # Resize images for faster processing
        images = [self._resize_image(img) for img in images]

        max_tokens = max_new_tokens or self.config.max_new_tokens

        # Build messages
        messages = self._build_messages(images, prompt)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decode only the generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    @torch.no_grad()
    def caption(
        self,
        image: Image.Image,
        detail_level: str = "normal",
    ) -> str:
        """
        Generate caption for a single image.

        Args:
            image: PIL Image to caption
            detail_level: "brief", "normal", or "detailed"

        Returns:
            Caption string
        """
        prompts = {
            "brief": "Describe this image briefly in one sentence.",
            "normal": "Describe what is happening in this image.",
            "detailed": "Provide a detailed description of this image, including all visible elements, actions, and any relevant context.",
        }

        prompt = prompts.get(detail_level, prompts["normal"])
        return self.generate(image, prompt, temperature=0.3, do_sample=False)

    @torch.no_grad()
    def compare_frames(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
    ) -> str:
        """
        Describe what changed between two video frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Description of changes/action
        """
        prompt = "What action or change is happening between these two frames?"
        return self.generate([frame1, frame2], prompt, temperature=0.3)

    @torch.no_grad()
    def analyze_sequence(
        self,
        frames: List[Image.Image],
        max_frames: int = 4,
    ) -> str:
        """
        Analyze a sequence of video frames.

        Args:
            frames: List of PIL Images (max 4 recommended)
            max_frames: Maximum frames to process

        Returns:
            Description of the sequence
        """
        frames = frames[:max_frames]
        prompt = "Describe the sequence of events shown in these frames."
        return self.generate(frames, prompt, temperature=0.3)

    @torch.no_grad()
    def answer_query(
        self,
        image: Image.Image,
        query: str,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: PIL Image
            query: Question about the image

        Returns:
            Answer string
        """
        return self.generate(image, query, temperature=0.5)

    @torch.no_grad()
    def extract_objects(
        self,
        image: Image.Image,
    ) -> List[str]:
        """
        List objects visible in the image.

        Args:
            image: PIL Image

        Returns:
            List of object names
        """
        prompt = "List all distinct objects visible in this image. Return only object names, separated by commas."
        response = self.generate(image, prompt, temperature=0.1, do_sample=False)

        # Parse comma-separated list
        objects = [obj.strip() for obj in response.split(",")]
        return [obj for obj in objects if obj]  # Filter empty strings

    @torch.no_grad()
    def reason_about_action(
        self,
        frames: List[Image.Image],
        action_query: str,
    ) -> Dict[str, str]:
        """
        Reason about an action across multiple frames.

        Args:
            frames: List of PIL Images showing action progression
            action_query: Query about the action (e.g., "What skill is being used?")

        Returns:
            Dictionary with reasoning results
        """
        frames = frames[:4]  # Limit to 4 frames

        results = {}

        # What is happening
        prompt = "What action or event is taking place across these frames?"
        results["action"] = self.generate(frames, prompt, temperature=0.3)

        # Answer specific query
        results["answer"] = self.generate(frames, action_query, temperature=0.5)

        return results

    @torch.no_grad()
    def generate_batch(
        self,
        batch_items: List[Dict],
        temperature: float = 0.3,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate responses for multiple image+prompt pairs using TRUE parallel batching.

        Each item is processed independently in parallel on the GPU, NOT concatenated
        into a single mega-prompt. This is much faster and more reliable.

        Args:
            batch_items: List of dicts with 'images' (List[Image]) and 'prompt' (str)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens per response

        Returns:
            List of generated responses (same order as input)
        """
        if not self._loaded:
            self.load()

        if not batch_items:
            return []

        max_tokens = max_new_tokens or self.config.max_new_tokens

        # Prepare all inputs independently
        all_texts = []
        all_images_list = []

        for item in batch_items:
            images = item['images']
            prompt = item['prompt']

            # Resize images
            images = [self._resize_image(img) for img in images]
            all_images_list.append(images)

            # Build message for this single item
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            all_texts.append(text)

        # Flatten all images for processor
        flat_images = []
        for images in all_images_list:
            flat_images.extend(images)

        # Set left padding for correct batched generation (decoder-only model)
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        # Process ALL inputs as a true batch
        inputs = self.processor(
            text=all_texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Restore original padding side
        self.processor.tokenizer.padding_side = original_padding_side

        # Generate ALL responses in parallel
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Decode each response independently
        responses = []
        for i in range(len(batch_items)):
            # Get the generated tokens for this item (excluding input)
            input_len = inputs.input_ids[i].shape[0]
            generated_ids = output_ids[i, input_len:]

            response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            responses.append(response.strip())

        return responses
