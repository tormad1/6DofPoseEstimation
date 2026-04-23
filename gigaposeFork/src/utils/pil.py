from PIL import Image


def open_image(path, inplane=None):
    image = Image.open(path)
    if inplane is not None:
        image = image.rotate(inplane)
    return image
