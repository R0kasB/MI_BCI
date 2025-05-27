
from PIL import Image, ImageDraw, ImageFont
import os

def combine_images(image_paths, output_path, layout='vertical', delete_original=True):
    """
    Combines multiple PNG images into a single image either stacked vertically or arranged horizontally.

    Args:
        image_paths (list): List of file paths to the PNG images to be combined.
        output_path (str): File path where the combined image will be saved.
        layout (str, optional): The layout for combining images. Options are 'vertical' or 'horizontal'.
                                Defaults to 'vertical'.

    Raises:
        ValueError: If the layout option is not 'vertical' or 'horizontal'.
        ValueError: If no valid images are found to combine.
    """
    # Filter valid images with extensions and check if the paths are not empty
    image_paths = [img for img in image_paths if os.path.isfile(img) and img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        raise ValueError("No valid images found to combine.")

    # Open all valid images and ensure they're in the correct mode
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGBA')  # Convert to RGBA to handle transparency correctly
            images.append(img)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    if not images:
        raise ValueError("No images could be opened.")

    if layout == 'vertical':
        # Calculate the max width and total height for vertical stacking
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        # Create a new blank image with the calculated dimensions
        combined_image = Image.new('RGBA', (max_width, total_height))

        # Paste images one below the other
        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height

    elif layout == 'horizontal':
        # Calculate the total width and max height for horizontal arrangement
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a new blank image with the calculated dimensions
        combined_image = Image.new('RGBA', (total_width, max_height))

        # Paste images side by side
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

    else:
        raise ValueError("Layout must be 'vertical' or 'horizontal'.")

    # Ensure the output path has a valid extension
    if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_path += '.png'  # Default to PNG if no extension is provided

    # Save the combined image, ensuring directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        combined_image.save(output_path)
        print(f"Combined image saved to {output_path}")

        if delete_original:
            # Delete original images after saving the combined image
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                    print(f"Deleted {img_path}")
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")

    except Exception as e:
        print(f"Error saving the combined image: {e}")


def add_border_and_title(image_path, output_path, title, border_size=10, title_height=40, title_size=20, 
                         border_color='black', title_color='white', title_bg_color='black'):
    """
    Adds a border around the image and a title on top of the image without overlapping the original image content.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image with border and title.
        title (str): Title text to add on top of the image.
        border_size (int, optional): Thickness of the border in pixels. Defaults to 10.
        title_height (int, optional): Height of the title area in pixels. Defaults to 40.
        title_size (int, optional): Font size of the title text. Defaults to 20.
        border_color (str, optional): Color of the border. Defaults to 'black'.
        title_color (str, optional): Color of the title text. Defaults to 'white'.
        title_bg_color (str, optional): Background color of the title area. Defaults to 'black'.
    """
    # Open the original image
    image = Image.open(image_path)

    # Calculate new image size including border and title space
    new_width = image.width + 2 * border_size
    new_height = image.height + border_size + title_height

    # Create a new image with expanded size and RGBA mode for transparency support
    new_image = Image.new('RGBA', (new_width, new_height), border_color)

    # Paste the original image onto the new canvas
    new_image.paste(image, (border_size, title_height))

    # Create a drawing context
    draw = ImageDraw.Draw(new_image)

    # Draw the title background rectangle
    draw.rectangle([(0, 0), (new_width, title_height)], fill=title_bg_color)

    # Load a font with the specified title size
    try:
        font = ImageFont.truetype("arial.ttf", title_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if "arial.ttf" is not available

    # Calculate text size and position for centering using textbbox
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (new_width - text_width) // 2
    text_y = (title_height - text_height) // 2

    # Draw the title text
    draw.text((text_x, text_y), title, fill=title_color, font=font)

    # Save the new image with border and title
    new_image.save(output_path)
    print(f"Image saved with border and title at {output_path}")
