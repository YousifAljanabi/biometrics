#!/usr/bin/env python3
"""
Convert all BMP files in training/reconstructors/u-net/data directory to PNG format.
This script will:
1. Find all .bmp files recursively in the data directory
2. Convert each BMP file to PNG format using PIL
3. Save the PNG file with the same name but .png extension
4. Optionally remove the original BMP files after successful conversion
"""

import os
import glob
from PIL import Image
import sys
from pathlib import Path

def convert_bmp_to_png(data_dir, remove_originals=False):
    """
    Convert all BMP files in the specified directory to PNG format.
    
    Args:
        data_dir (str): Path to the data directory containing BMP files
        remove_originals (bool): Whether to remove original BMP files after conversion
    
    Returns:
        tuple: (successful_conversions, failed_conversions)
    """
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return 0, 0
    
    # Find all BMP files recursively
    bmp_pattern = os.path.join(data_dir, "**", "*.bmp")
    bmp_files = glob.glob(bmp_pattern, recursive=True)
    
    if not bmp_files:
        print(f"No BMP files found in {data_dir}")
        return 0, 0
    
    print(f"Found {len(bmp_files)} BMP files to convert...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for bmp_file in bmp_files:
        try:
            # Create PNG filename by replacing .bmp extension with .png
            png_file = bmp_file.rsplit('.', 1)[0] + '.png'
            
            # Open and convert the image
            with Image.open(bmp_file) as img:
                # Convert to RGB if necessary (BMP files might be in different modes)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PNG
                img.save(png_file, 'PNG')
            
            print(f"✓ Converted: {os.path.basename(bmp_file)} -> {os.path.basename(png_file)}")
            successful_conversions += 1
            
            # Remove original BMP file if requested
            if remove_originals:
                os.remove(bmp_file)
                print(f"  Removed original: {os.path.basename(bmp_file)}")
                
        except Exception as e:
            print(f"✗ Failed to convert {bmp_file}: {str(e)}")
            failed_conversions += 1
    
    return successful_conversions, failed_conversions

def main():
    # Define the data directory path
    data_dir = "/home/yousifalaa/PycharmProjects/biometrics/training/reconstructors/u-net/data"
    
    print("BMP to PNG Converter")
    print("=" * 50)
    print(f"Target directory: {data_dir}")
    print()
    
    # First, convert all files without removing originals
    print("Step 1: Converting BMP files to PNG...")
    successful, failed = convert_bmp_to_png(data_dir, remove_originals=False)
    
    print()
    print("Conversion Summary:")
    print(f"✓ Successfully converted: {successful} files")
    print(f"✗ Failed conversions: {failed} files")
    
    if failed > 0:
        print("\nSome conversions failed. Please check the errors above.")
        print("Original BMP files have been preserved.")
        return 1
    
    if successful > 0:
        print("\nAll conversions completed successfully!")
        
        # Ask user if they want to remove original BMP files
        response = input("\nDo you want to remove the original BMP files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("\nStep 2: Removing original BMP files...")
            
            # Find and remove all BMP files
            bmp_pattern = os.path.join(data_dir, "**", "*.bmp")
            bmp_files = glob.glob(bmp_pattern, recursive=True)
            
            removed_count = 0
            for bmp_file in bmp_files:
                try:
                    os.remove(bmp_file)
                    removed_count += 1
                    print(f"✓ Removed: {os.path.basename(bmp_file)}")
                except Exception as e:
                    print(f"✗ Failed to remove {bmp_file}: {str(e)}")
            
            print(f"\nRemoved {removed_count} original BMP files.")
        else:
            print("\nOriginal BMP files have been preserved.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
