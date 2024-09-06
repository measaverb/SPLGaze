import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from utils.transforms import get_transforms


class MPIIFaceGazePerSubject(Dataset):
    def __init__(self, root, subject="p00", transform=None):
        if transform is None:
            transform = get_transforms()
        self.root = root
        self.subject = subject
        self.transform = transform

        self.lines = []
        with open(os.path.join(self.root, "Label", self.subject) + ".label") as f:
            self.lines = f.readlines()
        self.lines.pop(0)

        self.transforms = transform

    def parse_line(self, line):
        anno = {}
        anno["face"] = line[0]
        anno["name"] = line[3]
        anno["gaze_angle"] = line[7]
        return anno

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")
        anno = self.parse_line(line)

        img = Image.open(os.path.join(self.root, "Image", anno["face"]))
        img = self.transforms(img)

        gaze_angle = np.array(anno["gaze_angle"].split(",")).astype("float")
        gaze_angle = torch.from_numpy(gaze_angle).type(torch.FloatTensor)

        data = {}
        data["img"] = img
        data["gaze_angle"] = gaze_angle

        return data

    def __len__(self):
        return len(self.lines)


def get_multiple_subjects(root, subjects=("p00", "p01"), transform=None):
    datasets = []
    for subject in subjects:
        datasets.append(
            MPIIFaceGazePerSubject(root=root, subject=subject, transform=transform)
        )
    return ConcatDataset(datasets)


if __name__ == "__main__":
    ds = MPIIFaceGazePerSubject(root="/mnt/ssd-1tb/MPIIFaceGaze/", subject="p00")
    print(ds.__getitem__(0))

    ds = get_multiple_subjects(
        root="/mnt/ssd-1tb/MPIIFaceGaze/", subjects=("p00", "p01", "p02")
    )
    print(ds.__getitem__(0))
