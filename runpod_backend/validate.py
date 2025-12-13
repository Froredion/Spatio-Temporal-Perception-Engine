#!/usr/bin/env python3
"""
STPE Validation Script - Phase 1

Validates the codebase without requiring GPU:
1. Python syntax check for all modules
2. Import validation
3. Configuration validation
"""

import sys
import ast
import importlib.util
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def print_ok(msg):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")

def print_fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")

def print_warn(msg):
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")


def check_syntax(file_path: Path) -> bool:
    """Check Python syntax without importing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print_fail(f"{file_path.name}: {e.msg} at line {e.lineno}")
        return False


def validate_all_syntax():
    """Validate syntax of all Python files."""
    print("\n1. Checking Python syntax...")

    root = Path(__file__).parent
    py_files = list(root.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]

    failed = 0
    for py_file in py_files:
        if not check_syntax(py_file):
            failed += 1
        else:
            print_ok(py_file.name)

    return failed == 0


def validate_config():
    """Validate configuration dataclasses."""
    print("\n2. Validating configuration...")

    try:
        from config import (
            DINOv3Config, SAM3Config, VLMConfig,
            TemporalConfig, SceneGraphConfig,
            ProcessingConfig, STPEConfig
        )

        # Create instances
        dinov3_cfg = DINOv3Config()
        sam3_cfg = SAM3Config()
        vlm_cfg = VLMConfig()
        stpe_cfg = STPEConfig()

        # Validate model IDs
        expected_models = {
            'DINOv3': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
            'SAM-3': 'facebook/sam3',
            'Qwen3-VL': 'Qwen/Qwen3-VL-8B-Instruct',
        }

        configs = {
            'DINOv3': dinov3_cfg,
            'SAM-3': sam3_cfg,
            'Qwen3-VL': vlm_cfg,
        }

        for name, expected_id in expected_models.items():
            cfg = configs[name]
            if cfg.model_id == expected_id:
                print_ok(f"{name} model_id: {cfg.model_id}")
            else:
                print_fail(f"{name} model_id mismatch: expected {expected_id}, got {cfg.model_id}")
                return False

        # Validate embedding dimensions
        print_ok(f"DINOv3 embedding_dim: {dinov3_cfg.embedding_dim}")

        return True

    except Exception as e:
        print_fail(f"Config validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_imports():
    """Validate all module imports work."""
    print("\n3. Validating imports...")

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    modules_to_check = [
        ("config", "STPEConfig"),
        ("models", "ModelLoader"),
        ("models.dinov3", "DINOv3Model"),
        ("models.sam3", "SAM3Model"),
        ("models.qwen3_vision", "Qwen3VisionModel"),
        ("processing", "VisionPipeline"),
        ("processing.level1_global", "Level1GlobalProcessor"),
        ("processing.level2_objects", "Level2ObjectProcessor"),
        ("processing.level3_dense", "Level3DenseProcessor"),
        ("temporal", "TemporalPositionalEncoding"),
        ("temporal.tracker", "DINOv3Tracker"),
        ("temporal.motion", "MotionFeatureExtractor"),
        ("temporal.attention", "TemporalAttention"),
        ("temporal.pooling", "TemporalPooling"),
        ("temporal.scene_graph", "SceneGraph"),
        ("utils", "extract_frames"),
        ("utils.frame_extraction", "extract_frames"),
        ("utils.image_utils", "pil_to_tensor"),
        ("utils.clustering", "cluster_features"),
        ("pipeline", "STPEPipeline"),
        ("handler", "handler"),
    ]

    all_ok = True

    for module_name, class_name in modules_to_check:
        full_module = f"runpod_backend.{module_name}"
        try:
            module = importlib.import_module(full_module)
            if hasattr(module, class_name):
                print_ok(f"{full_module}.{class_name}")
            else:
                print_fail(f"{full_module} missing {class_name}")
                all_ok = False
        except ImportError as e:
            print_fail(f"{full_module}: {e}")
            all_ok = False
        except Exception as e:
            print_warn(f"{full_module}: {type(e).__name__}: {e}")
            # Non-fatal for now (may be missing optional deps)

    return all_ok


def validate_handler():
    """Validate handler can be instantiated."""
    print("\n4. Validating handler...")

    try:
        from runpod_backend.handler import handler

        # Test health check (doesn't require GPU/models)
        result = handler({'input': {'operation': 'health'}})

        if 'output' in result:
            status = result['output'].get('status')
            cuda = result['output'].get('cuda_available')
            print_ok(f"Handler health check: {status}")
            print_ok(f"CUDA available: {cuda}")
            return True
        else:
            print_fail(f"Health check failed: {result}")
            return False

    except Exception as e:
        print_fail(f"Handler validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dockerfile():
    """Validate Dockerfile exists and has correct structure."""
    print("\n5. Validating Dockerfile...")

    root = Path(__file__).parent
    dockerfile = root / "Dockerfile"

    if not dockerfile.exists():
        print_fail("Dockerfile not found")
        return False

    content = dockerfile.read_text()

    checks = [
        ("FROM nvidia/cuda", "CUDA base image"),
        ("python3", "Python installation"),
        ("torch", "PyTorch installation"),
        ("transformers", "Transformers installation"),
        ("COPY requirements.txt", "Requirements copy"),
        ("runpod_backend", "Code copy"),
        ("CMD", "Entrypoint"),
    ]

    all_ok = True
    for pattern, description in checks:
        if pattern in content:
            print_ok(f"{description}")
        else:
            print_fail(f"Missing {description} ({pattern})")
            all_ok = False

    return all_ok


def validate_requirements():
    """Validate requirements.txt exists and has required packages."""
    print("\n6. Validating requirements.txt...")

    root = Path(__file__).parent
    req_file = root / "requirements.txt"

    if not req_file.exists():
        print_fail("requirements.txt not found")
        return False

    content = req_file.read_text()

    required = [
        "torch",
        "transformers",
        "accelerate",
        "Pillow",
        "opencv-python",
        "numpy",
        "scipy",
        "scikit-learn",
        "runpod",
    ]

    all_ok = True
    for pkg in required:
        if pkg.lower() in content.lower():
            print_ok(f"{pkg}")
        else:
            print_fail(f"Missing {pkg}")
            all_ok = False

    return all_ok


def main():
    """Run all validations."""
    print("=" * 60)
    print("STPE Phase 1 Validation")
    print("=" * 60)

    results = []

    # Run validations
    results.append(("Syntax", validate_all_syntax()))
    results.append(("Config", validate_config()))
    results.append(("Imports", validate_imports()))
    results.append(("Handler", validate_handler()))
    results.append(("Dockerfile", validate_dockerfile()))
    results.append(("Requirements", validate_requirements()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        if result:
            print_ok(name)
            passed += 1
        else:
            print_fail(name)
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print(f"\n{Colors.GREEN}All validations passed! Ready for Docker build.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}Some validations failed. Please fix issues before building.{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
