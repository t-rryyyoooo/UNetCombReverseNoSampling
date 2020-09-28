from thinPatchCreater import ThinPatchCreater
from extractor import Extractor
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
from functions import croppingForNumpy
from pathlib import Path

class FeatureMapCreater():
    def __init__(self, image, model, image_patch_width, label_patch_width, image_patch_size, label_patch_size, mask, is_label = False):
        self.image = image
        self.model = model
        self.image_patch_width = image_patch_width # For ThinPatchCreater
        self.label_patch_width = label_patch_width # For ThinPatchCreater
        self.image_patch_size = image_patch_size # For this class
        self.label_patch_size = label_patch_size # For this class
        self.mask = mask
        self.is_label = is_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    1. Make 512-512-8 patch.
    2. Feed them to trained model and get feature maps.(64-512-512-8)
    3. Restore image from feature maps.
    4. Use extractor and get 48-48-32 patch.
    """

    def execute(self):
        """ Make feature map with model. """
        feature_map_list = self.makeFeatureMap()

        """ Crop feature map to image_patch_size. """
        dummy = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)
        for ch in range(self.num_channel):
            etr = Extractor(
                    image = feature_map_list[ch],
                    label = dummy, 
                    mask = self.mask,
                    image_patch_size = self.image_patch_size,
                    label_patch_size = self.label_patch_size
                    )

            etr.execute()
            
            feature_map_list[ch], _ = etr.output(kind="Array")

        feature_map_list = np.array(feature_map_list)
        length = feature_map_list.shape[1]
        self.feature_map_list = []
        for l in range(length):
            self.feature_map_list.append(feature_map_list[:, l, ...])

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

    def makeFeatureMap(self):
        """ Make 512-512-8 patch. """
        dummy = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)
        tpc = ThinPatchCreater(
                image = self.image,
                label = dummy, 
                image_patch_width = self.image_patch_width, 
                label_patch_width = self.label_patch_width,
                )

        tpc.execute()
        image_array_list, _ = tpc.output(kind="Array")

        """ Check the number of channel for feature map."""
        feature_map = self.getFeatureMapFromModel(image_array_list[0])
        self.num_channel = feature_map.shape[0]

        """ Caluculate crop size. """
        lower_z_crop_size = (self.image_patch_width - self.label_patch_width) // 2
        upper_z_crop_size = (self.image_patch_width - self.label_patch_width + 1) // 2
        lower_crop_size = np.array([0, lower_z_crop_size, 0, 0])
        upper_crop_size = np.array([0, upper_z_crop_size, 0, 0])

        """ Feed image_array_list to trained model. """
        feature_map_list = [[None] for _ in range(self.num_channel)]
        with tqdm(total=len(image_array_list), desc="Making feature maps...", ncols=60) as pbar:
            for image_array in image_array_list:
                feature_map = self.getFeatureMapFromModel(image_array)

                feature_map = croppingForNumpy(feature_map, lower_crop_size, upper_crop_size)
                for ch in range(self.num_channel):
                    feature_map_list[ch].append(feature_map[ch, ...])
                
                pbar.update(1)

        for ch in range(self.num_channel):
            feature_map_list[ch] = tpc.restore(feature_map_list[ch])

        return feature_map_list



    def getFeatureMapFromModel(self, image_array):
        image_array = torch.from_numpy(image_array)[None, None, ...].to(self.device, dtype=torch.float)

        feature_map = self.model.forwardWithoutSegmentation(image_array)
        feature_map = feature_map.to("cpu").detach().numpy().astype(np.float)
        feature_map = np.squeeze(feature_map)

        return feature_map



