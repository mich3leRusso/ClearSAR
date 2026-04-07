from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io

def show_image_with_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    scores: list[float] | None = None,
    show_label: bool = True,
    return_image: bool = False
) -> Image.Image | None:

    print(image.size)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x, y, w, h = box

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        if show_label and scores is not None and i < len(scores):
            label_y = y - 4 if y > 12 else y + h + 12
            ax.text(
                x + 2, label_y,
                f"score: {scores[i]:.2f}",
                color='white',
                fontsize=8,
                fontweight='bold',
                va='bottom',
                bbox=dict(facecolor='red', alpha=0.75, edgecolor='none', pad=2)
            )

    plt.axis('off')

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()  # .copy() detaches from the buffer before it closes

    plt.show()
    return None