from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image_with_boxes(image: Image.Image, boxes: list[list[float]]):
    """
    Display an image with bounding boxes overlaid.

    This function visualizes an image with rectangular bounding boxes drawn on top.

    Args:
        image (PIL.Image.Image): The image to display.
        boxes (list[list[float]]): List of bounding boxes in COCO format [x, y, w, h],
                                   where (x, y) is the top-left corner and (w, h) are width and height.
    """
    # Create a single subplot and show the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw each bounding box as a red rectangle
    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    # Hide axis ticks/labels for a cleaner visualization
    plt.axis('off')
    plt.show()
