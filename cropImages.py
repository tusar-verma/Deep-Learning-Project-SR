#!/usr/bin/env python3
"""
Image Resizing Script

This script takes a directory of images and creates resized versions while maintaining
the original aspect ratio. The new width is min(481, W) and height is calculated
proportionally to preserve the aspect ratio.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def calculate_resize_dimensions(original_width, original_height):
    """
    Calculate the resize dimensions while maintaining aspect ratio.
    New width is min(481, W) and height is calculated proportionally.
    
    Args:
        original_width: Original image width
        original_height: Original image height
    
    Returns:
        Tuple of (new_width, new_height)
    """
    new_width = min(481, original_width)
    
    # Calculate new height maintaining aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    
    return new_width, new_height

def process_image(input_path, output_path):
    """
    Process a single image: load, resize while maintaining aspect ratio, and save.
    
    Args:
        input_path: Path to the input image
        output_path: Path where the resized image will be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(input_path) as img:
            original_width, original_height = img.size
            new_width, new_height = calculate_resize_dimensions(original_width, original_height)
            
            # Skip if the image is already the correct size or smaller
            if original_width <= new_width:
                logger.info(f"Skipping {input_path.name}: already smaller than or equal to target width")
                # Just copy the original image
                img.save(output_path, quality=95, optimize=True)
                return True
            
            # Resize the image while maintaining aspect ratio
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            resized_img.save(output_path, quality=95, optimize=True)
            
            logger.info(f"Processed {input_path.name}: {original_width}x{original_height} -> {new_width}x{new_height}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """
    Process all images in the input directory and save resized versions to output directory.
    
    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for resized images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file_path)
    
    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        logger.info(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_file in image_files:
        output_file = output_path / image_file.name
        
        if process_image(image_file, output_file):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Resize images while maintaining aspect ratio. New width is min(481,W), height calculated proportionally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cropImages.py input_folder output_folder
  python cropImages.py /path/to/images /path/to/resized_images
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory containing input images'
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save resized images'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Resize formula: width = min(481, W), height calculated to maintain aspect ratio")
    
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()