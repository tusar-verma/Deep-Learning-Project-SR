import cv2
import sys

def downsample_image(input_path, output_path, scale):
    # Read the high-resolution image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    # Calculate new size
    height, width = img.shape[:2]
    new_width = int(width / scale)
    new_height = int(height / scale)

    # Downsample using bicubic interpolation
    lr_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Save the low-resolution image
    cv2.imwrite(output_path, lr_img)
    print(f"Saved downsampled image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python downsample.py <input_hr_image> <output_lr_image> <scale_factor>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale = float(sys.argv[3])
    downsample_image(input_path, output_path, scale)