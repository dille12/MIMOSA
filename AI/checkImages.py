import os


for x,y  in [["TEST_train_images", "TEST_validation_images"], ["train_images", "validate_images"]]:
    p = "AI/" + x
    p2 = "AI/" + y

    files1 = os.listdir(p)
    files2 = os.listdir(p2)
    i = 0
    for z in files1:
        if z in files2:
            #print(z)
            pass
        else:
            print("Not present", z)

        i += 1

    print("\n")
    print("FOLDER:", x, y)
    print("Total images", i)

