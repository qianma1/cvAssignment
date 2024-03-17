import glob
import imageio
from skimage.io.tests.test_mpl_imshow import plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class myImageDataset(Dataset):
    def __init__(self, inputs_root, labels_root):
        self.files = sorted(glob.glob(f"{inputs_root}/*.png"))
        self.files_using = sorted(glob.glob(f"{labels_root}/*.png"))

    def __getitem__(self, index):
        inputs = plt.imread(self.files[index % len(self.files)])
        labels = np.array(Image.open(self.files_using[index % len(self.files_using)]))
        return inputs, labels

    def __len__(self):
        return len(self.files)
