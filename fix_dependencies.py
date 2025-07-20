#!/usr/bin/env python3
"""
Fix Dependencies Script

This script helps fix NumPy compatibility issues that may occur when running
the video similarity learning project.

Usage:
    python fix_dependencies.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔧 Fixing NumPy compatibility issues...")
    print("This script will downgrade NumPy to a compatible version and reinstall dependencies.")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️  Warning: You're not in a virtual environment.")
        print("   Consider creating one first: python -m venv venv && source venv/bin/activate")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 1: Uninstall current NumPy
    if not run_command("pip uninstall numpy -y", "Uninstalling current NumPy"):
        return
    
    # Step 2: Install compatible NumPy version
    if not run_command("pip install 'numpy<2.0.0'", "Installing compatible NumPy version"):
        return
    
    # Step 3: Reinstall torch and torchvision to ensure compatibility
    if not run_command("pip install --force-reinstall torch torchvision torchaudio", "Reinstalling PyTorch"):
        return
    
    # Step 4: Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing project requirements"):
        return
    
    print("\n🎉 Dependency fix completed!")
    print("You should now be able to run the download_data.py script.")
    print("\nTry running: python scripts/download_data.py")

if __name__ == "__main__":
    main() 