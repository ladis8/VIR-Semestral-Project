import time
import pickle
import os.path
import numpy as np
import cv2
from cam_control import CamControl


IMAGES_GATHER = 10
SAVE_AS_PNG = True


class GRDataset:

    def __init__(self, name, res=(128,128), saveImages = False):
        self.name = name
        self.res = res
        self.saveImages = saveImages

        print("Checking for existing dataset...")
        filename = "data/{}".format(self.name)
        if os.path.exists(filename):
            self.dataset = pickle.load(open(filename, "rb"))
            print("Found existing dict with {} examples.".format(len(self.dataset)))
        else:
            print("No dataset found")
            self.dataset = []


    def gather(self, N, CAM, startindex= 0):
        assert N > 0; print("You specified zero examples to be gathered.")

        print("Gathering hand...")

        for i in range(N):
            img = CAM.render(self.res)
            self.dataset.append(img)
            if self.saveImages: cv2.imwrite("data/images/dataset_image{}.png".format(startindex + i), img)

        print("Saving dataset...")
        pickle.dump(self.dataset, open("data/{}".format(self.name), "wb"))


    def getBatch(self, N):
        """

        :param N: int, batchsize
        :return: [ndarray], Returns list of N normalized float images with shape Nxhxw 
        """
        ims = []

        for _ in range(N):
            img = self.dataset[np.random.randint(len(self.dataset))]
            img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            ims.append(img)

        ims = np.array(ims)
        return ims

if __name__ == "__main__":

    CAM = CamControl()
    dataset = GRDataset("gr_training_set", res=(128,128), saveImages= SAVE_AS_PNG)

    print("Get ready...")
    time.sleep(1)
    print("Gathering training dataset...")
    dataset.gather(IMAGES_GATHER, CAM, len(dataset.dataset))



