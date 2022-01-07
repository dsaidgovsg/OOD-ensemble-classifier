import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pyreadr
from pyreadr import pyreadr as pyr
import pandas as pd
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255),
                         (63.0/255, 62.1/255.0, 66.7/255.0)),
])

#root = "/Preliminary_Image_Dataset"

#onlyfiles = [f for f in os.listdir(root)]  # Get names of only the files, ignoring DS_Store and README
#onlyfiles.remove(".DS_Store")
#onlyfiles.remove("README.txt")

class PreliminaryImageDataset(Dataset):
    def __init__(self, transform = None):
        self.samples = []
        root = "/Preliminary_Image_Dataset"
        onlyfiles = [f for f in os.listdir(root)]  # Get names of only the files, ignoring DS_Store and README
        onlyfiles.remove(".DS_Store")
        onlyfiles.remove("README.txt")
        for i, classes in enumerate(onlyfiles):
            classes = os.path.join(root, classes)
            for j, images in enumerate(os.listdir(classes)):
                img = np.array(Image.open(os.path.join(classes, images)))
                self.samples.append([img, i])
        self.samples = np.array(self.samples, dtype='object')
        np.random.seed(3)
        p = np.random.permutation(len(self.samples))  # Designate 800 images to be test data, 400 of which is val
        self.transformations = transform
        self.test_data = self.samples[p[:800]][:, 0]
        self.test_labels = self.samples[p[:800]][:, 1]
        np.random.seed(3)
        p1 = np.random.permutation(len(self.test_data))
        self.test_data = self.test_data[p1[400:]]
        self.test_labels = [self.test_labels[i] for i in p1.tolist()[400:]]

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        if self.transformations is not None:
            img = self.transformations(img)
        #img = Image.fromarray(img)

        #plt.imshow(  img.permute(1, 2, 0)  )
        #plt.show()
        return img, label

    def __len__(self):
        return len(self.test_data)

#gdata = PreliminaryImageDataset()
#print(gdata[0])
#cifardata = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)


def agencyencoding(name):
    if name == 'NEA':
        return 0
    elif name == 'PUB':
        return 1
    elif name == 'TC':
        return 2
    elif name == 'AVA':
        return 3
    elif name == 'LTA':
        return 4
    elif name == 'NParks':
        return 5
    elif name == 'Others':
        return 6
    elif name == 'HDB':
        return 7
    elif name == 'SPF':
        return 8
    elif name == 'BCA':
        return 9
    elif name == 'MSO':
        return 10
    elif name == 'URA':
        return 11

"""
class MSODataset(Dataset):
    def __init__(self, img_path, label_path):
        self.samples = []
        self.transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(32, 32)),
                                                   transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                                        (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))])
        data = pyr.read_r(label_path)
        for key, val in data.items():
            df = val
        i = 0  # Limit the number of samples due to hardware limitations

        for subdir, dirs, files in os.walk(img_path):
            for file in files:
                idxfile = file.split("_")[0]
                print("idxfile:", idxfile)
                print(len(df.loc[df['Case Cover ID'] == idxfile].index))
                if len(df.loc[
                           df['Case Cover ID'] == idxfile].index) != 0:  # Check if image name has an entry in the table
                    print(os.path.join(subdir, file))
                    img = Image.open(os.path.join(subdir, file))

                    label = agencyencoding(df.loc[df["Case Cover ID"] == idxfile, "COA"].values[0])
                    print(label)
                    self.samples.append((img, label))
                    # img.close()
                    i += 1
                    print(i)
                if i == 7000:
                    return

    def __getitem__(self, index):
        img, label = self.samples[index]
        img = self.transformations(img)
        # plt.imshow(img.permute(1,2,0))
        # plt.show()
        return (img, label)

    def __len__(self):
        return len(self.samples)

imgpath = "D:/2018_Case_Image_Dataset/2018_Case_Image_Dataset"
labelpath = "D:/2018_case_dataset/data/mobile_data.rds"

if __name__ == "__main__":
    gdata = MSODataset(imgpath, labelpath)
    print(gdata[0][1])

"""
"""
import pickle
import pandas as pd
data = pickle.load(open(f"./results/pid_Imagenet_1.p", "rb"))
data = pd.read_pickle(f"./results/pid_Imagenet_1.p")
print(len(data['out_sfx']))
print(len(data['out_pro']))
"""