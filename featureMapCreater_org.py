import numpy as np
from labelPatchCreater import LabelPatchCreater
import torch
from extractor import Extractor
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

class FeatureMapCreater():
    def __init__(self, image, model, patch_size_for_model, plane_size, image_patch_size=[48, 48, 32], label_patch_size=[44, 44, 28], mask=None, is_label=False):
        self.image = image
        self.model = model
        self.patch_size_for_model = np.array(patch_size_for_model)
        self.plane_size = np.array(plane_size)
        self.image_patch_size = np.array(image_patch_size)
        self.label_patch_size = np.array(label_patch_size)
        self.mask = mask
        self.is_label = False

    def makeFeatureMap(self):
        """ Stand LabelPatchCreater to make patch for thin-UNet. """
        lpc = LabelPatchCreater(
                label = self.image, 
                patch_size = self.patch_size_for_model, 
                plane_size = self.plane_size, 
                is_label = self.is_label
                )

        lpc.execute()
        image_array_list = lpc.output(kind="Array")

        """ If we can use GPU. """
        use_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device)
        self.model.eval()

        """ Confirm the number of channel for model output. """
        image_array = torch.from_numpy(image_array_list[0])[None, None, ...].to(device, dtype=torch.float)
        predicted_array = self.model.forwardWithoutSegmentation(image_array)
        predicted_array = predicted_array.to("cpu").detach().numpy().astype(np.float)
        predicted_array = np.squeeze(predicted_array)

        self.num_channel = predicted_array.shape[0]

        predicted_array_list = [[None] for _ in range(self.num_channel)]
        with tqdm(total=len(image_array_list), desc="Making feature maps...", ncols=60) as pbar:
            for image_array in image_array_list:
                image_array = torch.from_numpy(image_array)[None, None, ...].to(device, dtype=torch.float)

                predicted_array = self.model.forwardWithoutSegmentation(image_array)
                predicted_array = predicted_array.to("cpu").detach().numpy().astype(np.float)
                predicted_array = np.squeeze(predicted_array)

                for ch in range(self.num_channel):
                    predicted_array_list[ch].append(predicted_array[ch, ...])
                pbar.update(1)

        for ch in range(self.num_channel):
            predicted_array_list[ch] = lpc.restore(predicted_array_list[ch])

        return predicted_array_list

    def execute(self):
        predicted_array_list = self.makeFeatureMap()

        label = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)

        for ch in range(self.num_channel):
            etr = Extractor(
                    image = predicted_array_list[ch],
                    label = label,
                    mask = self.mask,
                    image_patch_size = self.image_patch_size,
                    label_patch_size = self.label_patch_size
                    )

            etr.execute()

            predicted_array_list[ch], _ = etr.output(kind="Array")

        predicted_array_list = np.array(predicted_array_list)

        num_len = predicted_array_list.shape[1]
        self.feature_map_list= []
        for l in range(num_len):
            self.feature_map_list.append(predicted_array_list[:, l, ...])

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





