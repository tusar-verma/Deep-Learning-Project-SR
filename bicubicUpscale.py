from PIL import Image
import os
import sys

def upscale_bicubic(input_path, output_path, scale_factor):
    """
    Upscales an image using bicubic interpolation.

    Args:
        input_path (str): Path to the input low-res image.
        output_path (str): Path to save the upscaled image.
        scale_factor (int): Upscaling factor (e.g., 2, 3, 4).
    """
    # Load the image
    img = Image.open(input_path)
    
    # Get original dimensions
    original_width, original_height = img.size
    
    # Calculate new dimensions and convert to integers
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    new_size = (new_width, new_height)
    
    print(f"Original size: {original_width}x{original_height}")
    print(f"New size: {new_width}x{new_height}")
    
    # Resize using bicubic interpolation
    img_upscaled = img.resize(new_size, Image.BICUBIC)
    
    # Save the upscaled image
    img_upscaled.save(output_path)
    print(f"Image saved to: {output_path}")

def main():
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print("Usage: python3 bicubirUpscale.py <input_file> <output_file> <scale_factor>")
        print("Example: python3 bicubirUpscale.py in.png out.png 2")
        print("Scale factor should be a positive number (e.g., 2, 4, 0.5)")
        sys.exit(1)
    
    # Get input and output file paths from command line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse and validate scale factor
    try:
        scale_factor = float(sys.argv[3])
        if scale_factor <= 0:
            print("Error: Scale factor must be a positive number.")
            sys.exit(1)
    except ValueError:
        print("Error: Scale factor must be a valid number.")
        print("Examples: 2 (double size), 4 (quadruple size), 0.5 (half size)")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Check if input file has valid image extension
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    if not any(input_file.lower().endswith(ext) for ext in valid_extensions):
        print(f"Error: '{input_file}' does not appear to be a valid image file.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing: {input_file} -> {output_file}")
    print(f"Scale factor: {scale_factor}x")
    
    upscale_bicubic(input_file, output_file, scale_factor)
    
    print(f"Upscaling completed: {output_file}")

if __name__ == "__main__":
    main()