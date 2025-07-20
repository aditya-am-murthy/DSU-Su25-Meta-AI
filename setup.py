#!/usr/bin/env python3
"""
Setup script for the Video Similarity Learning project.
This script initializes the project structure and downloads sample data.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main setup function."""
    print("=== Video Similarity Learning - Project Setup ===")
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # Create necessary directories
    directories = [
        "data/videos",
        "data/models", 
        "checkpoints",
        "configs",
        "logs"
    ]
    
    print("\nCreating project directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Check if requirements are installed
    print("\nChecking dependencies...")
    try:
        import torch
        import torchvision
        import opencv_python
        import wandb
        import jupyter
        print("  ✓ All required packages are installed")
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        print("\nInstalling requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("  ✓ Requirements installed successfully")
        except subprocess.CalledProcessError:
            print("  ✗ Failed to install requirements")
            print("  Please run: pip install -r requirements.txt")
            return
    
    # Download sample data
    print("\nSetting up sample data...")
    try:
        subprocess.check_call([sys.executable, "scripts/download_data.py"])
        print("  ✓ Sample data created successfully")
    except subprocess.CalledProcessError:
        print("  ✗ Failed to create sample data")
        print("  Please run: python scripts/download_data.py")
        return
    
    # Create a simple test script
    test_script = project_root / "test_setup.py"
    with open(test_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Test script to verify the setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        from utils.video_utils import load_video
        from utils.data_utils import VideoDataset
        from utils.model_utils import create_model
        from utils.training_utils import ContrastiveLoss
        print("✓ All utility modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data():
    """Test if sample data exists."""
    data_dir = project_root / "data" / "videos"
    if data_dir.exists():
        video_files = list(data_dir.glob("*.mp4"))
        metadata_file = data_dir / "sample_metadata.csv"
        pairs_file = data_dir / "similarity_pairs.csv"
        
        if video_files and metadata_file.exists() and pairs_file.exists():
            print(f"✓ Sample data found: {len(video_files)} videos")
            return True
        else:
            print("✗ Sample data incomplete")
            return False
    else:
        print("✗ Data directory not found")
        return False

def test_model():
    """Test if model can be created."""
    try:
        model = create_model(
            model_type='similarity',
            backbone='resnet50',
            feature_dim=512,
            num_frames=30
        )
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Setup ===")
    
    tests = [
        ("Imports", test_imports),
        ("Data", test_data),
        ("Model", test_model)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\\nTesting {test_name}...")
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\\n🎉 All tests passed! Setup is complete.")
        print("\\nNext steps:")
        print("1. Start with notebook 01_Data_Exploration.ipynb")
        print("2. Run training: python scripts/train.py")
        print("3. View experiments: https://wandb.ai")
    else:
        print("\\n❌ Some tests failed. Please check the setup.")
''')
    
    print("  ✓ Test script created")
    
    # Make scripts executable
    scripts = ["scripts/download_data.py", "scripts/train.py", "test_setup.py"]
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            script_path.chmod(0o755)
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Run the test script: python test_setup.py")
    print("2. Start with the notebooks in the notebooks/ directory")
    print("3. Run training: python scripts/train.py --help")
    print("\nHappy learning! 🚀")

if __name__ == "__main__":
    main() 