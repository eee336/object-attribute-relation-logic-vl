from PIL import Image, ImageDraw

from oarlvla.gridworld.web_assets import remove_background_and_crop


def test_remove_background_and_crop_makes_transparent_cutout():
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 25, 135, 115), fill=(210, 20, 30))

    cutout, metrics = remove_background_and_crop(image, output_size=96)

    assert cutout.mode == "RGBA"
    assert cutout.size == (96, 96)
    assert cutout.getpixel((0, 0))[3] == 0
    assert max(cutout.getchannel("A").getextrema()) > 200
    assert 0.05 < metrics["mask_area_ratio"] < 0.6
