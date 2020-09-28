import argparse
import SimpleITK as sitk
import re
from functions import getSizeFromString
from featureMapCreater import FeatureMapCreater
import cloudpickle

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")
    parser.add_argument("model_path")
    parser.add_argument("save_path")
    parser.add_argument("--image_patch_size", default="48-48-32")
    parser.add_argument("--label_patch_size", default="44-44-28")
    parser.add_argument("--image_patch_width", default=8, type=int)
    parser.add_argument("--label_patch_width", default=8, type=int)
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--is_label", action="store_true")

    args = parser.parse_args()

    return args

def main(args):
    image = sitk.ReadImage(args.image_path)

    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    with open(args.model_path, "rb") as f:
        model = cloudpickle.load(f)

    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)

    fmc = FeatureMapCreater(
            image = image,
            model = model,
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            image_patch_size = image_patch_size,
            label_patch_size = label_patch_size,
            mask = mask,
            is_label = args.is_label
            )

    fmc.execute()
    fmc.save(args.save_path)

if __name__ == "__main__":
    args = parseArgs()
    main(args)


