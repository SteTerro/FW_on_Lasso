
import os
import re
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
"""
Dataset description: COIL-20 (Columbia Object Image Library)
transform it into a matrix X (flattened pixels, grayscale) and a vector y (exposure angle, degrees) 
for a LASSO regression problem with p >> n.

Each object has 72 images, one every 5 degrees of rotation (0-355)
Our choice: we use ALL images of a single object (default: obj1),
obtaining n=72 naturally, without the need for random sampling.
"""


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "coil-20")

OBJECT_ID = 1            # quale oggetto usare (1-20). Le immagini attese
                          # sono "obj{OBJECT_ID}__<angolo>.png"
IMG_SIZE = 32             # ridimensioniamo ogni immagine a IMG_SIZE x IMG_SIZE
                          # -> p = IMG_SIZE * IMG_SIZE (scala di grigi, 1 canale)


def _parse_angle_from_filename(filename: str, object_id: int):
    """
    Extract the exposure angle from the name of the file COIL-20.
    Expected format: obj<object_id>__<angolo>.png
    Return None if the file doesnt belong the request object or if the name
    doesnt respect the expected format
    """
    basename = os.path.basename(filename)
    match = re.match(rf"^obj{object_id}__(\d+)\.png$", basename)
    if match is None:
        return None
    return int(match.group(1))


def load_coil20(
    object_id: int = OBJECT_ID,
    img_size: int = IMG_SIZE,
    test_size: float = 0.3,
    standardize: bool = True,
    random_state: int = 42,
):
    """
    Loads all images of a single COIL-20 object, converts them to
    grayscale, resizes them, flattens them into pixel vectors, and
    constructs X (feature = pixel) and y (target = exposure angle in degrees).

    Returns a dict with X_train, X_test, y_train, y_test, n, p, object_id.
    """
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Folder doesn't found: {IMAGES_DIR}\n"
            "Make sure the COIL-20 images are extracted in "
            "'data/coil-20/' next to this script."
        )

    all_files = glob.glob(os.path.join(IMAGES_DIR, f"obj{object_id}__*.png"))
    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No image found for obj{object_id} in {IMAGES_DIR}. "
            "Check the file name format (expected: 'obj<N>__<angolo>.png')."
        )

    print(f"Fuond {len(all_files)} images for obj{object_id}.")

    X_list = []
    y_list = []
    skipped = 0

    for filepath in sorted(all_files):
        angle = _parse_angle_from_filename(filepath, object_id)
        if angle is None:
            skipped += 1
            continue

        img = Image.open(filepath).convert("L").resize((img_size, img_size))
        pixel_vector = np.asarray(img, dtype=np.float64).flatten()  # p = img_size^2

        X_list.append(pixel_vector)
        y_list.append(angle)

    if skipped > 0:
        print(f"Warning: {skipped} files discarded because their names don't conform to the expected format.")

    X = np.vstack(X_list)                    # shape (n, p)
    y = np.array(y_list, dtype=np.float64)   # shape (n,), degree angle

    n, p = X.shape
    print(f"Constructed dataset: n={n} images, p={p} pixel (ratio p/n = {p/n:.2f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if standardize:
        mu_X, sigma_X = X_train.mean(axis=0), X_train.std(axis=0)
        sigma_X[sigma_X == 0] = 1.0  # constant pixel (e.g. uniform background)
        X_train = (X_train - mu_X)/sigma_X
        X_test = (X_test - mu_X)/sigma_X

        mu_y, sigma_y = y_train.mean(), y_train.std()
        if sigma_y == 0:
            sigma_X = 1
        y_train = (y_train - mu_y)/sigma_y
        y_test = (y_test - mu_y)/sigma_y

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n": n,
        "p": p,
        "object_id": object_id,
        "img_size": img_size,
    }


if __name__ == "__main__":
    data = load_coil20()
    print("\nShape training set:", data["X_train"].shape)
    print("Shape test set:    ", data["X_test"].shape)
    print(
        "Min/max angle in the training set (AFTER centering, therefore as a discard "
        "from the mean, not absolute angle):",
        data["y_train"].min(), data["y_train"].max()
    )

"""
Nota:
Given an image of this specific object, we understand how its pixels predict the angle 
from which it was photographed.

We don't generalize across different objects; this is a specific problem confined to that 
single object, which varies only with the shooting angle (same background, same lighting, 
same camera, same distance; the only variable is rotation).
"""