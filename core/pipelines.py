import time
import os
import pygame
import tifffile
from PIL import Image
import numpy as np



def cropImagesToDir(app):
    PATH = os.path.join(app.MAINPATH,"ALLIMAGES")
    print("Process images:", PATH)
    PATH_DST = os.path.join(app.MAINPATH,"PROCESSEDIMAGES")
    print(PATH_DST)
    for x in os.listdir(PATH):

        IMPATH = f"{PATH}/{x}"

        app.loadImage(image_path = IMPATH)
        time.sleep(0.2)
        app.EXPORT = True
        app.execCalc()
        while app.CALCING:
            time.sleep(0.1)

        im = pygame.surfarray.array3d(app.imageApplied)

        print(im.shape)
        im = np.transpose(im, (1, 0, 2))
        print(im.shape)
        print(im.dtype)
        print(np.max(im))

        im2 = Image.fromarray(im)
        im2.save(f"{PATH_DST}/{x}")
        #time.sleep(1)
        print("Pipeline moves to next image")

        