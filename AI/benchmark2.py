import os
import traceback
models = os.listdir("NEURAL NETWORKS/")
pickedModels = []
os.environ['SM_FRAMEWORK'] = 'tf.keras'
for i,x in enumerate(models):
    
    for y in ["ITER17"]:
        if y in x:
            pickedModels.append(x)
            break



try:
    #MODEL = models[int(i)-1]
    from benchmarker.runtests import main

    for x in pickedModels:

        if "ITER14" in x or "ITER16" in x or "ITER17" in x:
            RGB = True
        else:
            RGB = False

        print("Beginning benchmark of:", x)
        print("RGB:", RGB)

        main(x, RGB)

except:
    traceback.print_exc()
    pass



