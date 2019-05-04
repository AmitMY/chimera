import io

import numpy as np
from PIL import Image


def get_file_bytes(file, format="PNG"):
    if isinstance(file, str):
        f = open(file, "rb")
        b = f.read()
        f.close()

        return b

    if isinstance(file, bytearray) or isinstance(file, bytes):
        return file

    if isinstance(file, np.ndarray):
        image = Image.fromarray(file)

    buffer = io.BytesIO()
    image.save(buffer, format=format)

    return buffer.getvalue()
