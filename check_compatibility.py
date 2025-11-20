"""Compatibility check script for RDT package"""

import sys
import importlib.util
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)"

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if package is installed and optionally check version"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, f"✗ {package_name} not installed"
        
        # Try to import and get version
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            if min_version:
                # Simple version comparison (not perfect but good enough)
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    return True, f"✓ {package_name} {version}"
                else:
                    return False, f"✗ {package_name} {version} (requires >= {min_version})"
            return True, f"✓ {package_name} {version}"
        else:
            return True, f"✓ {package_name} (version unknown)"
    except ImportError as e:
        return False, f"✗ {package_name} import failed: {e}"
    except Exception as e:
        return False, f"✗ {package_name} check failed: {e}"

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            return True, f"✓ CUDA {cuda_version} available ({device_count} device(s), {device_name})"
        else:
            return False, "✗ CUDA not available (CPU only)"
    except:
        return False, "✗ Cannot check CUDA"

def check_mps() -> Tuple[bool, str]:
    """Check Apple MPS availability"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "✓ Apple MPS available"
        else:
            return False, "✗ Apple MPS not available"
    except:
        return False, "✗ Cannot check MPS"

def check_package_structure() -> List[Tuple[bool, str]]:
    """Check if RDT package structure is correct"""
    checks = []
    
    try:
        import rdt
        checks.append((True, f"✓ rdt package found (v{rdt.__version__})"))
        
        # Check main modules
        modules = ['model', 'data', 'trainer', 'utils']
        for module_name in modules:
            try:
                module = importlib.import_module(f'rdt.{module_name}')
                checks.append((True, f"✓ rdt.{module_name} importable"))
            except ImportError as e:
                checks.append((False, f"✗ rdt.{module_name} import failed: {e}"))
        
        # Check if main classes are accessible
        try:
            from rdt import RDT, WikiTextDataset, RDTTrainer
            checks.append((True, "✓ Main classes accessible (RDT, WikiTextDataset, RDTTrainer)"))
        except ImportError as e:
            checks.append((False, f"✗ Main classes import failed: {e}"))
        
        # Check scripts
        try:
            from rdt.scripts import train, inference
            checks.append((True, "✓ Scripts accessible (train, inference)"))
        except ImportError as e:
            checks.append((False, f"✗ Scripts import failed: {e}"))
        
    except ImportError:
        checks.append((False, "✗ rdt package not found (not installed?)"))
    
    return checks

def main():
    print("="*60)
    print("RDT Package Compatibility Check")
    print("="*60)
    
    all_passed = True
    
    # Python version
    print("\n1. Python Version")
    passed, msg = check_python_version()
    print(f"   {msg}")
    all_passed &= passed
    
    # Core dependencies
    print("\n2. Core Dependencies")
    dependencies = [
        ('torch', '2.0.0'),
        ('transformers', '4.30.0'),
        ('datasets', '2.14.0'),
        ('yaml', None),  # pyyaml
        ('tensorboard', '2.13.0'),
        ('tqdm', '4.65.0'),
        ('numpy', '1.24.0'),
    ]
    
    for package, min_ver in dependencies:
        passed, msg = check_package(package, min_ver)
        print(f"   {msg}")
        all_passed &= passed
    
    # Check packaging (optional)
    print("\n3. Packaging Tools (optional)")
    for package in ['packaging', 'setuptools', 'wheel']:
        passed, msg = check_package(package)
        print(f"   {msg}")
        # Don't fail on optional packages
    
    # Hardware acceleration
    print("\n4. Hardware Acceleration")
    cuda_passed, cuda_msg = check_cuda()
    print(f"   {cuda_msg}")
    
    mps_passed, mps_msg = check_mps()
    print(f"   {mps_msg}")
    
    if not cuda_passed and not mps_passed:
        print("   ⚠ No GPU acceleration available (will use CPU)")
    
    # Package structure
    print("\n5. Package Structure")
    structure_checks = check_package_structure()
    for passed, msg in structure_checks:
        print(f"   {msg}")
        all_passed &= passed
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! RDT is ready to use.")
        print("\nNext steps:")
        print("  - Train: rdt-train --config rdt/configs/base.yaml")
        print("  - Test: python test_model.py")
        print("  - Docs: see README.md and USAGE_GUIDE.md")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("\n  pip install -e .")
        print("  or")
        print("  pip install rdt-transformer")
        print("\nFor detailed installation instructions, see INSTALL.md")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    try:
        # Try to import packaging for version comparison
        import packaging.version
    except ImportError:
        print("Warning: 'packaging' module not found. Version checks will be skipped.")
        print("Install with: pip install packaging\n")
    
    sys.exit(main())
