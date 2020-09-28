import argparse
import SimpleITK as sitk
import re
from functions import getSizeFromString
from featureMapCreaterV2 import FeatureMapCreater
import cloudpickle

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")
    parser.add_argument("model_path")
    parser.add_argument("save_path")
    parser.add_argument("--image_patch_size", default="48-48-32")
    parser.add_argument("--label_patch_size", default="44-44-28")
    parser.add_argument("--padding_size", default="1-1-1")
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
    padding_size = getSizeFromString(args.padding_size)

    fmc = FeatureMapCreater(
            image = image,
            model = model,
            image_patch_size = image_patch_size,
            label_patch_size = label_patch_size,
            mask = mask,
            padding_size = padding_size,
            is_label = args.is_label
            )

    fmc.execute()
    fmc.save(args.save_path)

if __name__ == "__main__":
    args = parseArgs()
    main(args)


