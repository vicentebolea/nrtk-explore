from PIL.Image import Image
from PIL import Image as ImageModule

import base64
import io
import random
import warnings


def image_to_base64_str(img: Image, format: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return f"data:image/{format};base64," + base64.b64encode(buf.getvalue()).decode()


class ImagesManager:
    def __init__(self, local_state=None):
        if local_state:
            local_state["images_cache"] = {}
            self.images = local_state["images_cache"]
        else:
            self.images = {}

    def LoadImage(self, path):
        if path not in self.images:
            img = ImageModule.open(path)
            img = img.resize((224, 224))
            img = img.convert("RGB")
            self.images[path] = img

        return self.images[path]

    def ComputeBase64(self, img_id, img):
        return image_to_base64_str(img, "png")
