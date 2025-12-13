"""
SAM-3 Model Wrapper (Client)

Communicates with the SAM-3 server running in a dedicated conda environment.
The server runs in the 'sam3' conda env with PyTorch 2.7 / CUDA 12.6.

This wrapper can be imported from the main RAPIDS environment and sends
requests to the SAM-3 server via socket.
"""

import os
import base64
import socket
import struct
import pickle
import subprocess
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image


SAM3_SERVER_HOST = "127.0.0.1"
SAM3_SERVER_PORT = 9999


class SAM3Model:
    """SAM-3 client that communicates with the SAM-3 server."""

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.host = SAM3_SERVER_HOST
        self.port = SAM3_SERVER_PORT
        self._server_process = None
        self._loaded = False

    def load(self):
        """Connect to SAM-3 server (should be started by start.sh)."""
        if self._loaded:
            return

        # Wait for server to be ready (started by start.sh)
        max_wait = 120  # seconds
        start_time = time.time()
        print("Waiting for SAM-3 server to be ready...")

        while time.time() - start_time < max_wait:
            if self._check_server():
                print("SAM-3 server is ready")
                self._loaded = True
                return
            time.sleep(2)
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"  Still waiting for SAM-3 server... ({elapsed}s)")

        raise RuntimeError(
            "SAM-3 server not available. Make sure start.sh was run first, "
            "or check /workspace/sam3_server.log for errors."
        )

    def _start_server(self):
        """Start the SAM-3 server process."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_script = os.path.join(script_dir, "sam3_server.py")

        # Use conda run to execute in sam3 environment
        cmd = [
            "conda", "run", "-n", "sam3", "--no-capture-output",
            "python", server_script,
            "--host", self.host,
            "--port", str(self.port),
            "--device", self.device,
        ]

        # Start server in background
        log_file = "/workspace/sam3_server.log"
        with open(log_file, "w") as log:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        print(f"SAM-3 server starting (PID: {self._server_process.pid})")
        print(f"Logs: {log_file}")

    def _check_server(self) -> bool:
        """Check if server is running and responding."""
        try:
            response = self._send_request({"method": "health"}, timeout=5)
            return response.get("status") == "ok"
        except Exception:
            return False

    def unload(self):
        """Stop the SAM-3 server."""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

        self._loaded = False

    def _send_request(self, request: dict, timeout: float = 60) -> dict:
        """Send request to SAM-3 server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((self.host, self.port))

            # Send request
            data = pickle.dumps(request)
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise RuntimeError("No response from server")
            msg_length = struct.unpack(">I", length_data)[0]

            # Receive response
            response_data = b""
            while len(response_data) < msg_length:
                chunk = sock.recv(min(msg_length - len(response_data), 65536))
                if not chunk:
                    break
                response_data += chunk

            return pickle.loads(response_data)

        finally:
            sock.close()

    def _image_to_b64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

    def segment_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects matching a text description.

        Args:
            image: PIL Image to segment
            text_prompt: Text description of objects to find
            threshold: Detection confidence threshold

        Returns:
            List of dicts with mask, bbox, score
        """
        if not self._loaded:
            self.load()

        request = {
            "method": "segment_with_text",
            "image_b64": self._image_to_b64(image),
            "text_prompt": text_prompt,
            "threshold": threshold or self.config.default_threshold,
        }

        response = self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"SAM-3 error: {response['error']}")

        return self._decode_segments(response.get("segments", []))

    def segment_automatic(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Automatically segment all salient objects.

        Args:
            image: PIL Image to segment
            threshold: Detection confidence threshold

        Returns:
            List of dicts with mask, bbox, score
        """
        if not self._loaded:
            self.load()

        request = {
            "method": "segment_automatic",
            "image_b64": self._image_to_b64(image),
            "threshold": threshold or self.config.default_threshold,
        }

        response = self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"SAM-3 error: {response['error']}")

        return self._decode_segments(response.get("segments", []))

    def segment_with_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment object within a bounding box.

        Args:
            image: PIL Image
            box: (x1, y1, x2, y2) bounding box
            threshold: Detection confidence threshold

        Returns:
            List of dicts with mask, bbox, score
        """
        if not self._loaded:
            self.load()

        request = {
            "method": "segment_with_box",
            "image_b64": self._image_to_b64(image),
            "box": list(box),
            "threshold": threshold or self.config.default_threshold,
        }

        response = self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"SAM-3 error: {response['error']}")

        return self._decode_segments(response.get("segments", []))

    def _decode_segments(self, segments: List[dict]) -> List[Dict[str, Any]]:
        """Decode segments from server response."""
        decoded = []

        for seg in segments:
            # Decode mask from base64
            mask_bytes = base64.b64decode(seg["mask_b64"])
            mask_shape = tuple(seg["mask_shape"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)

            decoded.append({
                "mask": mask,
                "bbox": tuple(seg["bbox"]),
                "score": seg["score"],
                "label": seg.get("label", "object"),
            })

        return decoded
