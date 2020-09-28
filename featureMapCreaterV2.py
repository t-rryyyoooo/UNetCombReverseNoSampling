from extractor import Extractor
import numpy as np
import SimpleITK as sitk
from functions import croppingForNumpy
from pathlib import Path
import torch
from tqdm import tqdm

class FeatureMapCreater():
    def __init__(self, image, model, image_patch_size, label_patch_size, mask=None, padding_size_in_model=[1, 1, 1], is_label=False):
        self.image = image
        self.model = model
        self.image_patch_size = np.array(image_patch_size)
        self.label_patch_size = np.array(label_patch_size)
        self.mask = mask
        self.padding_size_in_model = np.array(padding_size_in_model)
        self.is_label = is_label

    def execute(self):
        """ To make feature map correctly, add self.padding_size_in_model."""
        padded_image_patch_size = self.image_patch_size + self.padding_size_in_model * 2
        dummy = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)
        """ Make patch. """
        etr = Extractor(
                image = self.image, 
                label = dummy, 
                image_patch_size = padded_image_patch_size, 
                label_patch_size = self.label_patch_size, 
                mask = self.mask
                )

        etr.execute()
        image_array_list, _ = etr.output(kind="Array")

        is_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if is_cuda else "cpu")

        """ Make feature maps. """
        self.feature_map_list = []
        with tqdm(total=len(image_array_list), desc="Making feature maps..", ncols=60) as pbar:
            for image_array in image_array_list:
                image_array = torch.from_numpy(image_array)[None, None, ...].to(device, dtype=torch.float)

                feature_map = self.model.forwardWithoutSegmentation(image_array)
                feature_map = feature_map.to("cpu").detach().numpy().astype(np.float)
                feature_map = np.squeeze(feature_map)

                lower_crop_size = [0] + self.padding_size_in_model.tolist()
                upper_crop_size = [0] + self.padding_size_in_model.tolist()
                feature_map = croppingForNumpy(feature_map, lower_crop_size, upper_crop_size)
                self.feature_map_list.append(feature_map)

                pbar.update(1)

    def output(self):
        return self.feature_map_list

    def save(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.is_label:
            name = "label_"
        else:
            name = "image_"

        with tqdm(total=len(self.feature_map_list), desc="Saving feature maps...", ncols=60) as pbar:
            for i, feature_map in enumerate(self.feature_map_list):
                path = save_path / (name + str(i).zfill(4) + ".npy")
                np.save(str(path), feature_map)

                pbar.update(1)

