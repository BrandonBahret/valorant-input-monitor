from PIL import Image
import os

def generate_ico(png_path, ico_path):
    """Generate a multi-resolution .ico file from a PNG."""
    print(f"Generating {ico_path} from {png_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(ico_path), exist_ok=True)
    
    # Standard Windows icon sizes
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    # Load the source PNG
    img = Image.open(png_path)
    
    # Generate resized versions
    icon_images = []
    for size in sizes:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        icon_images.append(resized)
    
    # Save as .ico with all sizes
    icon_images[0].save(
        ico_path,
        format='ICO',
        sizes=[img.size for img in icon_images],
        append_images=icon_images[1:]
    )
    
    print(f"✓ Icon generated successfully with {len(sizes)} sizes")

if __name__ == "__main__":
    png_source = "assets/favicon-512x512.png"
    ico_output = "assets/favicon.ico"  # Changed to output to assets folder
    
    if not os.path.exists(png_source):
        print(f"ERROR: {png_source} not found!")
        exit(1)
    
    generate_ico(png_source, ico_output)