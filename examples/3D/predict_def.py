from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist3D
import argparse
from aicsimageio import AICSImage


def read_image(input_path):
    print("Reading image...")
    return imread(input_path)


def normalize_image(img):
    print("Normalizing image...")
    return normalize(img, 1, 99.8)


def crop_image(image, z_start, z_end, x_start, x_end, y_start, y_end):
    """
    Crop the image based on the provided start and end indices for z, x, and y dimensions.

    Parameters:
    - image: 3D NumPy array (z, x, y)
    - z_start, z_end: start and end indices for the z dimension
    - x_start, x_end: start and end indices for the x dimension
    - y_start, y_end: start and end indices for the y dimension

    Returns:
    - Cropped image as a 3D NumPy array
    """
    return image[z_start:z_end, x_start:x_end, y_start:y_end]


def predict(model_dir, img_normed, n_tiles):
    model = StarDist3D(None, name=model_dir, basedir='.')

    print("Weights loaded successfully")
    label_starfinity, res = model.predict_instances(img_normed, n_tiles=n_tiles,
                                                    affinity=True, affinity_thresh=0.1,
                                                    verbose=True)
    # label_starfinity, res = model.predict_instances_big(
    #     img_normed, axes='ZYX', block_size=10, min_overlap=2,
    #     n_tiles=n_tiles, affinity=True, affinity_thresh=0.1, verbose=True
    # )
    print(res.keys())
    return label_starfinity, res["markers"]


def save_results(outdir, input_path, label_starfinity, label_stardist):
    if outdir is not None:
        print("Saving results...")
        Path(outdir).mkdir(exist_ok=True, parents=True)
        basename = str(Path(outdir) / Path(input_path).stem)
    imwrite(f"{basename}.starfinity.tif", label_starfinity[:, np.newaxis].astype(np.uint16), imagej=True,
            compression='zlib')
    imwrite(f"{basename}.stardist.tif", label_stardist[:, np.newaxis].astype(np.uint16), imagej=True,
            compression='zlib')


def main(input_path, model_dir, outdir):
    img = read_image(input_path)
    # # Define crop indices
    # z_start, z_end = 10, 20
    # x_start, x_end = 30, 80
    # y_start, y_end = 50, 100
    #
    # # Crop the image
    # img = crop_image(img, z_start, z_end, x_start, x_end, y_start, y_end)

    n_tiles = tuple(int(np.ceil(s / 128)) for s in img.shape)
    img_normed = normalize_image(img)
    label_starfinity, label_stardist = predict(model_dir, img_normed, n_tiles)
    save_results(outdir, input_path, label_starfinity, label_stardist)
    print("Done")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run StarDist3D prediction on a given image')
    # parser.add_argument('-i', '--input', type=str, required=True, help="Input file path")
    # parser.add_argument('-m', '--model', type=str, required=True, help="Model directory")
    # parser.add_argument('-o', '--outdir', type=str, required=True, help="Output directory")
    # parser.add_argument('--n_tiles', type=int, nargs=3, default=[1, 1, 1], help="Tiling dimensions (x, y, z)")

    # args = parser.parse_args()
    # print(args)

    # main(args.input, args.model, args.outdir, args.n_tiles)
    # imgs = AICSImage("data/examples/LHA3_R5_tiny_V02.czi")
    # img = img = imgs.data[0][0][2]
    img = Path("codes/train/images/LHA_R3_13_1390_1178.tif")
    model = "my_model"
    outdir = Path("/local/workdir/stardist/codes/test/")
    main(img, model, outdir)
