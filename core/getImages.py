import os
import shutil


for x in os.listdir("E:/Old SEM images"):
    for y in os.listdir(f"E:/Old SEM images/{x}"):
        if ".tif" in y:
            src = f"E:/Old SEM images/{x}/{y}"
            dst = f"C:/Users/Reset/Documents/GitHub/ALLIMAGES/{y}"
            if os.path.exists(dst):
                continue
            shutil.copyfile(src, dst)
            print(y, "COPIED")