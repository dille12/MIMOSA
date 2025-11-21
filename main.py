from loadLock import *

LOADLOCKSTATE["load_point"] = "Importing libraries"


import numpy as np
import tifffile
from matplotlib import pyplot as plt
import pygame
import sys
from pygame.math import Vector2 as v2
from core.keypress import key_press_manager
from core.slider import Slider
LOADLOCKSTATE["load_point"] = "Importing Tensorflow"
from core.funcs.imageDisplay import ImageDisplayCache
from core.node import Node, InputNode
from core.numbaAccelerated import thresholdArrayJIT, sobel_edge_detection_jit, CannyDetection
from core.node import getNodes, Node
from core.gametick import GameTick
from core.menuNode import MenuNode
import time
from core.funcs.bezier import draw_bezier_curve
from tkinter import filedialog
import pickle
from core.nodeBluePrint import nodeBluePrint
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from pathlib import Path
from _thread import start_new_thread
from core.funcs.saveload import saveSetupClean, loadSetupClean
from utils.config import load_config
from AI.ai_core import load_model  
import traceback
import core.pipelines
import tensorflow as tf
from utils.jsonFetch import set_value, get_value
from PIL import Image
from scipy.spatial.distance import pdist
from skimage.measure import perimeter
from core.nodeUtilities.sliderIndependent import sliderIndependent
from core.button import Button
from dispPipeline import pipelineTick, Sample, fast_mean_feret_diameter
from AI.loadCustomModel import load_custom_segmentation_model
from core.funcs.rectRenderer import rectRenderer
from core.keyhint import KeyHint
from core.configureMask import configureMask

LOADLOCKSTATE["load_point"] = "Imports Done!"

def get_max_diameter(label_array, label):
    """
    Computes the maximum diameter (feret diameter) of a given label.

    :param label_array: 3D numpy array containing labeled regions (H, W, C)
    :param label: The label number to analyze
    :return: Maximum diameter of the label (float)
    """
    # Extract the first channel assuming labels are stored there
    label_array_2d = label_array[..., 0]  # Take the first channel

    # Get coordinates of all pixels belonging to the label
    y_coords, x_coords = np.where(label_array_2d == label)

    if len(x_coords) == 0:
        return 0  # No pixels found for this label

    # Stack coordinates as (x, y) pairs
    coordinates = np.column_stack((x_coords, y_coords))

    # Compute pairwise distances between all label pixels
    pairwise_distances = pdist(coordinates)

    # Return the maximum distance found
    return np.max(pairwise_distances)

class Notification:
    def __init__(self, app: "App", text, delay = 0):
        self.app = app
        self.text = text
        self.lifetime = -delay
        self.maxLifetime = 4
        self.pos = v2([10,self.app.res[1]])
        self.posD = self.pos.copy()
        self.app.notifications.insert(0, self)
        if len(self.app.notifications) > 4:
            self.app.notifications.pop()

    def tick(self):
        self.lifetime += min(self.app.clock.get_time() / 1000.0, 1/30)

        if self.lifetime < 0:
            return

        if self.lifetime > self.maxLifetime:
            self.app.notifications.remove(self)
            return

        self.place = self.app.notifications.index(self)
        self.pos = v2([10, self.app.res[1] - 25 * (self.place + 1) - 25])
        self.posD = self.pos * 0.25 + self.posD * 0.75
        t = self.app.fontSmall.render(self.text, True, [255,255,255])

        if self.lifetime > self.maxLifetime - 1:
            alpha = int(255 * (self.maxLifetime - self.lifetime))
            t.set_alpha(alpha)

        self.app.screen.blit(t, self.posD)

    


class App(ImageDisplayCache):
    def __init__(self):
        self.keyhints = []
        print("Initializing app!")
        LOADLOCKSTATE["load_point"] = "Initializing app"
        pygame.init()
        super().__init__()
        self.notifications = []
        

        infoObject = pygame.display.Info()
        self.fullres = v2(infoObject.current_w, infoObject.current_h)
        self.res = v2(infoObject.current_w-100, infoObject.current_h-100)
        self.res_save = self.res.copy()
        self.MAINCOLOR = [255,165,0]
        
        self.fullscreen = False
        
        self.clock = pygame.time.Clock()
        self.zoom = 1
        self.zoomDelta = 1
        self.v2 = v2
        self.GT = GameTick

        self.tooltips = []
        self.TOOLTIP = None
        self.MAINPATH = os.getcwd()
        self.DATAPATH = self.MAINPATH + "/data"
        if not os.path.exists(self.DATAPATH):
            os.mkdir(self.DATAPATH)
        self.imPath = None
        self.NODES = []
        self.CURR_NODES = self.NODES

        self.VIGNETTE = pygame.image.load("assets/vignette.png").convert_alpha()

        self.THRESHOLD = sliderIndependent([20,20], self, 0, 10, 5, "Diameter threshold")
        self.ADJUSTX = sliderIndependent([20,700], self, -10, 10, 1, "ADJUSTX")
        self.ADJUSTY = sliderIndependent([20,860], self, -10, 10, 1, "ADJUSTY")

        self.mouseAvailable = True

        self.callbacks = {
            "configureMask": configureMask
        }
        self.editingMask = False
        self.maskRectOrigin = None
        
        
        self.DATAPATH = self.MAINPATH + "/data"
        self.lastIm = self.get_value("lastIm", "assets/D13_03.tif")
        if not os.path.exists(self.lastIm):
            self.lastIm = "assets/D13_03.tif"


        self.loadImage(self.lastIm)
        #self.applyLuminosityMask(threshold=0)
        self.setBackground(self.image.copy())
        self.keypress = []
        self.keypress_held_down = []

        self.restartSaveFile = os.path.join(os.getcwd(), "data/RESET.pkl")
        
        self.cameraPos = v2([1920/2,1080/2]) - v2(self.image.get_size())/2
        self.mouse_pos = v2([0,0])
        self.font = pygame.font.Font("assets/terminal.ttf", 30)
        self.fontSmall = pygame.font.Font("assets/terminal.ttf", 20)
        self.fontSmaller = pygame.font.Font("assets/terminal.ttf", 10)
        self.fontNode = pygame.font.Font("assets/terminal.ttf", 16)
        #self.fontNode.set_script(True)
        self.fontRoboto = pygame.font.Font("assets/Roboto-Regular.ttf", 20)
        self.imagePos = v2([0,0])
        self.luminositySlider = Slider(self, 0, 255, 128, [1920 - 220, 20], self.applyLuminosityMask, [])
        self.edgeDetectionSlider = Slider(self, 0, 100, 10, [1920 - 220, 60], self.edgeDetectionActivator, [])
        self.mode = 0
        self.modeX = 0


        self.CONFIG = load_config()

        self.nodeColorGrade = {
            "Math": 210,         # Cool blue for mathematical precision and logic
            "Thresholding": 40,  # Warm yellow for decision-making and cutoffs
            "Contrast": 20,      # Orange for enhancing distinctions
            "Algorithm": 0,      # Red for intensive, high-energy operations
            "AI": 90,
            "Filters": 160,      # Teal/green for transformation and image processing
            "Essential": 120,    # Deep blue for foundational, core utilities
            "Data": 280,         # Purple for abstraction and organization
            "Misc": 300,         # Magenta for miscellaneous, unpredictable tasks
        }

        LOADLOCKSTATE["load_point"] = "Building nodes"
        self.NODEBLUEPRINTS = getNodes(self, self)
        print(self.NODEBLUEPRINTS)
        self.activeInputNode = None
        self.clicked = False
        self.activeNode = False
        self.calcTick = self.GT(20)
        self.CALCING = False
        self.strI = 0
        self.connectionUnderMouse = None
        self.dT = 0
        self.CALCNODE = None
        self.EXPORT = False
        self.menunodes = []
        self.selRectOrigin = None
        self.notification = None
        self.selectionRect = None
        self.TRAINING = False
        print("MAIN PATH:", self.MAINPATH)
        self.modelLoaded = None

        self.PROCESSEDDATA = None
        self.PROCESSING = False
        self.processingProgress = ""
        
        print("DATA PATH", self.DATAPATH)
        print(os.path.exists(self.DATAPATH)) 

        
        self.separateDisplay = False
        self.dualWindow = None
        self.dualWindowSurf = None

        self.mMouseKH = KeyHint(self, "MIDDLE MOUSE", "Pan view")
        self.zoomKH = KeyHint(self, "SCROLL WHEEL", "Zoom")
        self.defineRenderWindowKH = KeyHint(self, "LCLICK (HOLD)", "Draw a render window")
        self.deleteRenderWindow = KeyHint(self, "R", "Delete render window")

        self.rightClickTick = 0


        self.contextMenuShow = []

        # Base directory: folder where the executable or script resides
        BASE_DIR = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)

        # Create subdirectories alongside the executable
        for folder in ["NEURALNETWORKS", "presets", "pipelines"]:
            path = os.path.join(BASE_DIR, folder)
            os.makedirs(path, exist_ok=True)

        LOADLOCKSTATE["load_point"] = "Loading semantic model"
        MODEL = "ITER17_SM_SHAPE_384x384x1_1.keras"
        if os.path.exists(f"{self.MAINPATH}/NEURALNETWORKS/{MODEL}"):
            self.MODEL = load_custom_segmentation_model(f"{self.MAINPATH}/NEURALNETWORKS/{MODEL}")



        LOADLOCKSTATE["load_point"] = "Loading segmentation model"
        MODEL = "ITER21_RETRAIN_765IMAGES_SHAPE_128x128x1_1.keras"
        if os.path.exists(f"{self.MAINPATH}/NEURALNETWORKS/{MODEL}"):
            self.MODELSEGM = load_custom_segmentation_model(f"{self.MAINPATH}/NEURALNETWORKS/{MODEL}")
        
        

        self.imageLoadDir = self.get_value("imageLoadDir", "")
        print(self.imageLoadDir)

        self.renderRect = None

        self.nodeCloseTick = self.GT(30, oneshot=True)
        self.AiProcessRect = None

        
        self.initMenus()

        self.buttonAddSample = Button(self, "Add new sample", [50,70], tooltip="Add a new sample to the batch. Each sample is processed separately.")
        self.buttonAddData = Button(self, "Load image batch", [50,105], tooltip="Load a batch of images to be processed. Images require a .txt file in tandem for metadata.")
        self.buttonSaveData = Button(self, "Save Pipeline", [50,140], tooltip="Saves the images and results.")
        self.buttonLoadData = Button(self, "Load Pipeline", [50,175], tooltip="Loads a pipeline with its images and results.")


        self.buttonRun = Button(self, "Run AI", [450,70], tooltip="Runs the currently loaded node structure on each image listed under samples not nicknamed UNSORTED.")
        self.buttonInspect = Button(self, "Inspect Results", [650,70], "Inspect each image after AI processing. Visually decide if results are of acceptable quality.") 
        self.buttonDisp = Button(self, "Calculate Dispersion", [1000,70], "When images are inspected, calculate the dispersion of each sample.")
        self.buttonGoBackInspect = Button(self, "Go back", [20,275])
        self.buttonToggleInspect = Button(self, "Toggle", [20,240], tooltip="Toggle between the original image and the AI processed image.")
        self.buttonUse = Button(self, "Use", [20,150], tooltip="Use the current image in dispersion analysis. Image is of acceptable quality.")
        self.buttonDontUse = Button(self, "Don't use", [20,185], tooltip="Don't use the current image in dispersion analysis. Image is of unacceptable quality.")

        self.buttonGoBackDisplay = Button(self, "Go back", [20,240])
        self.buttonToggleArea = Button(self, "Toggle Area", [20,275], tooltip="Toggle the graph between particle counts vs particle area sum.")

        self.exportCSV = Button(self, "Export CSV", [1000,105], tooltip="Toggle the graph between particle counts vs particle area sum.")


        
        self.darkenSurf = pygame.Surface(self.res, pygame.SRCALPHA).convert_alpha()
        self.darkenSurf.fill((0,0,0))
        self.darkenSurf.set_alpha(155)


        self.SAMPLEDISPLAY = None
        

        self.SAMPLEDATA = {"Unsorted": Sample(self, "Unsorted", 50)}
        self.pickedUp = None
        self.pipelineImage = None
        self.PIPELINERESULT = None
        self.pipelineInspect = False
        self.InspectImages = []
        self.PIPELINEACTIVE = False
        self.inspectToggle = False
        self.PIPELINEPROGRESS = 0
        self.PIPELINETARGETPROGRESS = 1
        self.PIPELINESUBPROGRESS = 0
        self.pipelineEstimate = 0
        self.pipelineTimeToComplete = 0
        self.progressSave = 0

        #Notification(self, "Welcome to MIMOSA!")

        if os.path.exists(self.restartSaveFile):
            loadSetupClean(self, self.CURR_NODES, self.restartSaveFile)

            #os.remove(self.restartSaveFile)


    def killScreen(self):
        self.sepScreenButton.text = "Separate Display"
        print("Closing separate display")
        self.dualWindow.destroy()
        self.separateDisplay = False
        self.dualWindow = None
        self.dualWindowSurf = None


    def loadConfig(self):
        self.CONFIG = load_config()

        
    def separateScreen(self):

        if self.separateDisplay:
            self.killScreen()
            return
            
        
        self.sepScreenButton.text = "Combine Display"
        print("Separating screen")
        self.separateDisplay = True
        self.dualWindow = pygame.Window(
            title="MIMOSA - Image", size = (800, 600), resizable=True)
        
        self.dualWindowSurf = self.dualWindow.get_surface()
        print(self.dualWindow)

    def initScreen(self):

        self.res = self.fullres - [100,100]
        self.res_save = self.res.copy()
        self.screen = pygame.display.set_mode(self.res, pygame.RESIZABLE)
        pygame.display.set_caption("MIMOSA")
        pygame.display.set_icon(pygame.image.load("assets/ICON.png").convert_alpha())

        self.rectRenderer = rectRenderer(self.MAINPATH + "/assets/ICONpixelmap.png")

        print("Screen initialized")


    def saveImageAs(self):
        pass
    
    def notify(self, text):
        Notification(self, text)
        print("Notification:", text)

    
    def refreshKeyHints(self):
        for hint in self.keyhints:
            hint.active = False

    def renderKeyHints(self):
        i = 0
        for hint in self.keyhints:
            if not hint.active:
                continue

            x = 20
            y = 300 + i * 30
            self.screen.blit(hint.surface, (x,y))
            i+=1


    def drawLogo(self):

        rSize = min(int(self.res[0] / 64), int(self.res[1] / 64))
        offset =  self.res/2 - [rSize*32, rSize*32]

        self.rectRenderer.change_rect_size(rSize)
        self.rectRenderer.render_optimized_surface(self.screen, offset[0], offset[1], color = [60,39,0])

    def initMenus(self):


        self.PrimeNodeMenu = MenuNode(self, pos = [20,20], text = "Add Node")
        self.MathMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Math")
        #self.ProcessingMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Processing")

        self.thresholdsMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Thresholding")
        self.contrast = MenuNode(self, parent = self.PrimeNodeMenu, text = "Contrast")
        self.algorithms = MenuNode(self, parent = self.PrimeNodeMenu, text = "Algorithm")
        self.AI = MenuNode(self, parent = self.PrimeNodeMenu, text = "AI")


        self.filters = MenuNode(self, parent = self.PrimeNodeMenu, text = "Filters")


        self.EssentialMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Essential")
        self.DataMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Data")
        self.MiscMenu = MenuNode(self, parent = self.PrimeNodeMenu, text = "Misc")

        for nodeText in self.NODEBLUEPRINTS:
            
            x = self.NODEBLUEPRINTS[nodeText]

            for y in self.menunodes:
                if y.text == x.nodeType:
                    MenuNode(self, parent=y, text = x.name, returnNode = x)
                    break

            else:
                MenuNode(self, parent=self.MiscMenu, text = x.name, returnNode = x)


        self.editMenu = MenuNode(self, sibling=self.PrimeNodeMenu, text = "Edit")
        self.setup = MenuNode(self, parent= self.editMenu, text="Setup")

        
        self.save = MenuNode(self, parent=self.setup, text="Save", returnFunc=lambda: saveSetupClean(self))
        self.load = MenuNode(self, parent=self.setup, text="Load", returnFunc=lambda: loadSetupClean(self, self.NODES))
        
        self.reset = MenuNode(self, parent=self.setup, text="Reset", returnFunc=self.clearSetup)
        
        

        self.imsMenu = MenuNode(self, sibling=self.editMenu, text = "Image")


        self.loadIm = MenuNode(self, parent=self.imsMenu, text="Load Image", returnFunc=self.loadImageDialog, tooltip="Load an image.")
        self.loadIm2 = MenuNode(self, parent=self.imsMenu, text="Load Next Image", returnFunc=self.loadAI, tooltip="Load a next untrained image.")
        self.saveIm = MenuNode(self, parent=self.imsMenu, text="Save image as", returnFunc=self.saveImageAs, tooltip="Save the exported output.")


        self.ais = MenuNode(self, parent=self.editMenu, text="Stage AI", returnFunc=self.stageAI, tooltip="Stage an neural network for use of KERAS_Model node.")
        self.pipelineMenu = MenuNode(self, parent=self.editMenu, text="Init Pipeline", returnFunc=self.initPipeline, tooltip="Initializes a pipeline, which computes all images in a specified folder.")
        self.resetAppMenu = MenuNode(self, parent=self.editMenu, text="Restart App", returnFunc=self.resetApp, tooltip="Reset the app with the current setup.")
        self.toggleAutoCalc = MenuNode(self, parent=self.editMenu, text="Disable AutoCalc", returnFunc=self.toggleCalc, tooltip="Toggles the automatic initiation of the calculation pipeline. Disable if using computationally intensive nodes.")
        MenuNode(self, parent=self.editMenu, text="Create stitch", returnFunc=self.stitch, tooltip="Creates a stitch of the original image and processed image.")
        self.sepScreenButton = MenuNode(self, parent=self.editMenu, text="Separate Display", returnFunc=self.separateScreen, tooltip="Creates a separate window to display the image on.")
        self.autoCalc = True

    def initPipeline(self):
        start_new_thread(core.pipelines.cropImagesToDir, (self, ))



    def stitch(self, output_dir="stitched_images", base_filename="stitched"):
        try:
            # Convert pygame surfaces to NumPy arrays
            image1_array = pygame.surfarray.array3d(self.image)
            image2_array = pygame.surfarray.array3d(self.imageApplied)

            # Swap axes from (width, height, channels) to (height, width, channels) for PIL
            image1_array = np.transpose(image1_array, (1, 0, 2))
            image2_array = np.transpose(image2_array, (1, 0, 2))

            # Convert NumPy arrays to PIL Images
            image1 = Image.fromarray(image1_array)
            image2 = Image.fromarray(image2_array)

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Find the height of the taller image
            height = max(image1.height, image2.height)

            # Resize images to have the same height
            image1 = image1.resize((int(image1.width * height / image1.height), height))
            image2 = image2.resize((int(image2.width * height / image2.height), height))

            # Calculate the total width of the combined image
            total_width = image1.width + image2.width

            # Create a new blank image with the appropriate size
            stitched_image = Image.new('RGB', (total_width, height))

            # Paste the two images side by side
            stitched_image.paste(image1, (0, 0))
            stitched_image.paste(image2, (image1.width, 0))

            # Generate a unique filename
            existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith(".png")]
            existing_numbers = [int(f[len(base_filename):-4]) for f in existing_files if f[len(base_filename):-4].isdigit()]
            next_number = max(existing_numbers) + 1 if existing_numbers else 1
            output_path = os.path.join(output_dir, f"{base_filename}{next_number}.png")

            # Save the stitched image
            stitched_image.save(output_path)
            print(f"Stitched image saved as {output_path}")
        except:
            traceback.print_exc()




    def toggleCalc(self):
        self.autoCalc = not self.autoCalc
        if self.autoCalc:
            self.toggleAutoCalc.text = "Disable AutoCalc"
        else:
            self.toggleAutoCalc.text = "Enable AutoCalc"

    def stageAI(self):
        try:
            
            start_new_thread(self.threadedAIstage, ())
            
        except Exception as e:
            Notification(self, str(e))

    def threadedAIstage(self):
        model_path = filedialog.askopenfilename(
            defaultextension=".keras",
            filetypes=[("KERAS models", "*.keras")],  # Only allow .pkl files
            title="Stage AI",
            initialdir=f"{self.MAINPATH}/NEURALNETWORKS")
        #model_path = "particle_segmentation_unet.keras"  # Path to the trained model
        self.notify("Loading AI model...")
        self.MODEL = load_custom_segmentation_model(model_path)
        #MenuNode(self, parent = self.MiscMenu, text = "KERAS_Model", returnNode=self.NODEBLUEPRINTS["KERAS_Model"])
        self.MODEL.summary()
        self.modelLoaded = os.path.splitext(os.path.basename(model_path))[0]

        self.clearCache()
        self.notify(f"Model {self.modelLoaded} loaded!")


    def get_value(self, key, default = None):
        return get_value(self.DATAPATH + "/data.json", key, default=default)
    
    def set_value(self, key, value):
        return set_value(self.DATAPATH + "/data.json", key, value)
    
    def resetApp(self):
        saveSetupClean(self, FILE=self.restartSaveFile)
        os.execv(sys.executable, ['python'] + sys.argv)


    def loadImageDialog(self):
        self.imageLoadDir = self.get_value("imageLoadDir", "")
        print("Initial directory:", self.imageLoadDir)
        f = filedialog.askopenfilename(
            title="Load image",
            initialdir=self.imageLoadDir)
        if f:
            folderPath = os.path.dirname(os.path.abspath(f))
            print("FOLDER OF IMAGE", folderPath)
            self.imageLoadDir = folderPath
            self.set_value("imageLoadDir", self.imageLoadDir)

            print("Gotten value from json:", self.get_value("imageLoadDir", "nothing"))

        else:
            return

        try:
            print("Loading image...")
            #imName = Path(f).stem
            #print(imName)
            #trainIms = os.listdir("AI/train_images")
            #print(trainIms)
            #trainImNames = " ".join(trainIms)
            #print(trainImNames)
            #if imName in trainImNames:
            #    print("IMAGE ALREADY PRESENT")
            #    self.notify("This image is already in training directory.")
            self.loadImage(image_path=f)
            
        except Exception as e:
            print(e)

    def loadAI(self):
        imPath = getUntrainedImages()[0]

        self.loadImage(image_path=imPath)

    def tickNotification(self, start = True):
        if self.notification:
            if not start:
                res = self.notification.tick()

                if res == 2:
                    self.notification = None

                elif res == 1:
                    self.notification = None


            else:
                self.notification.kp = self.keypress.copy()

                self.keypress.clear()
                self.keypress_held_down.clear()


        
    def clearSetup(self):
        self.NODES.clear()
        self.setBackground(self.image.copy())
        self.notify("Setup cleared!")


    def closeMenus(self):
        for x in self.menunodes:
            x.selected = False

    def tickToolTips(self):
        
        RENDERED = False

        for x in self.tooltips:
            if x.renderedLastTick:
                
                RENDERED = True
            else:   
                x.invokeTick.value = 0

            x.renderedLastTick = False

        if not RENDERED:
            self.TOOLTIP = None

        if self.TOOLTIP:
            self.screen.blit(self.TOOLTIP, self.mouse_pos + [5,5])

        

    def updateMenu(self):

        self.nodeUnderMouse = False

        for x in self.menunodes:
            x.render()

        if not self.nodeUnderMouse:
            if self.nodeCloseTick.tick():
                self.closeMenus()

        else:
            self.nodeCloseTick.value = 0

    def edgeDetectionActivator(self, val):
        
        image_array = pygame.surfarray.array3d(self.image)

        
        thresholded_image_array = CannyDetection(image_array, np.mean(image_array, axis=2), val)

        self.setBackground(pygame.surfarray.make_surface(thresholded_image_array))

    def renderSliders(self):
        if self.luminositySlider.update():
            self.applyLuminosityMask(threshold=self.luminositySlider.value)

        if self.edgeDetectionSlider.update():
            pass

    def applyLuminosityMask(self, threshold=0):
        # Get the numpy array from the surface
        image_array = pygame.surfarray.array3d(self.image)  # Shape (width, height, 3)

        thresholded_image_array = thresholdArrayJIT(image_array, np.mean(image_array, axis=2), threshold)

        self.setBackground(pygame.surfarray.make_surface(thresholded_image_array))

    def loadImage(self, image_path = "D13_03.tif"):
        # Load the TIFF image using tifffile
        self.imPath = image_path
        try:
            image_array = tifffile.imread(image_path)
            print("Loading image data...")
            with tifffile.TiffFile(image_path) as tif:
                volume = tif.asarray()
                axes = tif.series[0].axes
                imagej_metadata = tif.imagej_metadata
                print(imagej_metadata)
        except:
            print("Not a tiff file")
            im = pygame.image.load(image_path)
            image_array = pygame.surfarray.array3d(im)  
            image_array = image_array.transpose([1, 0, 2])

        self.set_value("lastIm", image_path)

        self.imageName = Path(image_path).stem
        print(self.imageName)


        # Normalize or scale the image data to match the format expected by pygame (8-bit per channel)
        # Ensure the numpy array has the proper dimensions for color or grayscale
        if image_array.dtype != np.uint8:
            image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

        # If the image is grayscale (single channel), convert it to RGB by repeating the channel
        if len(image_array.shape) == 2:  # Grayscale image
            image_array = np.stack([image_array] * 3, axis=-1)

        if image_array.shape[2] not in [1, 3]:
            image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)

        # Convert the numpy array to a Pygame Surface
        surface = pygame.surfarray.make_surface(np.transpose(image_array, (1, 0, 2)))

        # Optional: Convert the Surface to match the display format
        self.image = surface.convert()
        self.imageSize = v2(self.image.get_size())
        self.setBackground(self.image.copy())
        self.clearCache()
        self.notify("Image loaded!")

    def visualizeModel(self):

        if self.separateDisplay:
            SCREEN = self.dualWindowSurf
        else:
            SCREEN = self.screen

        if self.AiProcessRect:
            try:
                origin = v2(self.pixelToViewport(self.AiProcessRect[0:2]))
                s = v2(self.pixelToViewport(self.AiProcessRect[2:4]))
                r = pygame.Rect(origin, s - origin)
                pygame.draw.rect(SCREEN, [0,255,255], r, 1)
            except TypeError:
                print("Visualization rect removed during blit. Excepting.")
            except:
                traceback.print_exc()

    def toggleFullscreen(self):
        if "f11" in self.keypress:
            if self.fullscreen:
                self.screen = pygame.display.set_mode(self.res_save, pygame.RESIZABLE)
                self.fullscreen = False
            else:
                self.screen = pygame.display.set_mode(self.fullres, pygame.FULLSCREEN | pygame.NOFRAME)
                self.fullscreen = True
                self.res_save = self.res.copy()

    def loop(self):

        self.refreshKeyHints()

        

        self.mouseAvailable = True
        if self.fullscreen:
            self.res = v2(self.fullres)
        else:
            self.res = v2(self.screen.get_size())
        
        for x in self.tooltips:
            x.renderedLastTick = False



        self.screen.fill((0,0,0))

        if self.separateDisplay:
            self.dualWindowSurf.fill((0,0,0))

        self.mousePosDelta = self.mouse_pos.copy()
        key_press_manager(self)
        self.mouseDelta = self.mouse_pos - self.mousePosDelta

        

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit() 

            if event.type == pygame.WINDOWCLOSE:
                id = event.window.id
                print(f"WINDOWCLOSE event sent to Window #{id}.")
                self.killScreen()


        if "space" in self.keypress:
            self.mode += 1
            if self.mode == 3:
                self.mode = 0

        self.modeX = 0.8 * self.modeX + 0.2 * self.mode

        if self.separateDisplay or self.mode == 2:
            self.drawLogo()

        try:
            if self.mode in [0, 1] or self.separateDisplay:

                self.mMouseKH.active = True

                self.displayImageCached()
                self.drawRenderRect()
                self.visualizeModel()
            
        except:
            traceback.print_exc()
            sys.exit()

        self.toggleFullscreen()

        if self.mode == 1 or self.separateDisplay:
            #self.renderSl iders()
            self.zoomKH.active = True
            self.defineRenderWindowKH.active = True
            if self.renderRect:
                self.deleteRenderWindow.active = True
            self.handleMouseImPos()

            
            
        if self.mode == 0:
            self.updateNodes()
            self.updateMenu()  

        if self.mode == 2:
            pipelineTick(self)

        if self.editingMask:
            self.drawMaskingRect()
        else:
            self.drawSelectionRect()
            

        if self.autoCalc:
            self.execCalc() 
        elif "enter" in self.keypress:
            self.execCalc()
        
        self.renderKeyHints()

        self.tickToolTips()

        self.tickContextMenus()

        if "mouse2" in self.keypress_held_down:
            self.rightClickTick += 1
        else:
            self.rightClickTick = 0

        
        
        temp = self.notifications.copy()
        for x in temp:
            x.tick()

        


    def clearCache(self):
        for x in self.CURR_NODES:
            x.clearCache()


    def drawRenderRect(self):

        if self.separateDisplay:
            SCREEN = self.dualWindowSurf
        else:   
            SCREEN = self.screen

        if not self.renderRect:
            return
        
        p1, p2 = self.renderRect

        pixel1 = self.pixelToViewport(p1)
        pixel2 = self.pixelToViewport(p2)
        r = pygame.Rect(pixel1, pixel2-pixel1)
        pygame.draw.rect(SCREEN, [255,0,0], r, width = 1)
        t = self.fontSmaller.render("RENDER BOX (PRESS R TO REMOVE)", True, [255,0,0])
        SCREEN.blit(t, pixel1 + [0, -15])


    def notify(self, text):
        Notification(self, text)

    def applyMask(self, node):
        node.parent.clearCache()
        r, cache = node.parent.calc()
        node.parent.clearCache()

        #app.imageApplied = pygame.surfarray.make_surface(result[:,:,0:3].astype("uint8"))
        #app.setBackground(pygame.surfarray.make_surface(result))
        if not isinstance(node.value, np.ndarray) or node.value.shape != r.shape[0:2]:
            node.value = np.ones(r.shape[0:2], dtype=np.uint8)
            
        if len(r.shape) == 2:  # grayscale → RGB
            r = np.stack([r] * 3, axis=-1)

        mask = node.value.astype(np.float32)

        # Apply mask: unmasked (0) pixels become red, masked (1) pixels green
        r = r.astype(np.float32)
        r[:,:,0] *= (1 - mask)     # red channel → off where masked
        r[:,:,1] *= mask           # green channel → on where masked
        r[:,:,2] *= 0              # blue → off entirely

        r = np.clip(r, 0, 255).astype(np.uint8)
        self.setBackground(pygame.surfarray.make_surface(r))

        
    def drawMaskingRect(self):
        if self.separateDisplay:
            SCREEN = self.dualWindowSurf
        else:
            SCREEN = self.screen

        MODE_ON = self.mode == 1 or (self.separateDisplay and self.dualWindow.focused)

        if not MODE_ON or not self.editingMask:
            return
        
        if "i" in self.keypress:
            self.editingMask.value = 1 - self.editingMask.value
            self.applyMask(self.editingMask)

        if "esc" in self.keypress:
            self.editingMask = False
            self.maskRectOrigin = None
            self.maskingRect = None
            #self.applyMask(self.editingMask)
            self.EXPORT = True
            self.mode = 0
            return

        # Initialize origin for mouse0 or mouse2
        if ("mouse0" in self.keypress or "mouse2" in self.keypress) and not self.activeInputNode:
            self.maskRectOrigin = list(self.mouse_pos.copy()) + [0 if "mouse2" in self.keypress else 1]
            print("New pos at", self.maskRectOrigin)

        # Draw while mouse button is held
        if self.maskRectOrigin and ("mouse0" in self.keypress_held_down or "mouse2" in self.keypress_held_down):
            p = [min(self.maskRectOrigin[0], self.mouse_pos[0]),
                min(self.maskRectOrigin[1], self.mouse_pos[1])]
            s = [abs(self.maskRectOrigin[0] - self.mouse_pos[0]),
                abs(self.maskRectOrigin[1] - self.mouse_pos[1])]
            self.maskingRect = pygame.Rect(p, s)
            pygame.draw.rect(SCREEN, [0, 255, 0] if "mouse0" in self.keypress_held_down else [255,0,0], self.maskingRect, width=1)  # green outline

                

        # Finalize rectangle
        else:
            if self.maskRectOrigin:
                print("Finalizing masking rect:", self.maskingRect)
                p = self.viewportToPixel(self.maskingRect.topleft)
                s = self.viewportToPixel(self.maskingRect.bottomright) - p
                # Apply to mask
                x0, y0 = p
                x1, y1 = x0 + s[0], y0 + s[1]

                # Clip to mask dimensions
                mask_w, mask_h = self.editingMask.value.shape
                x0, x1 = max(0, x0), min(mask_w, x1)
                y0, y1 = max(0, y0), min(mask_h, y1)


                if x1 > x0 and y1 > y0:
                    self.editingMask.value[int(x0):int(x1), int(y0):int(y1)] = self.maskRectOrigin[2]
                    print("APPLYING MASK IN", x0, y0, x1, y1, self.maskRectOrigin[2])
                    self.applyMask(self.editingMask)


            self.maskRectOrigin = None
            self.maskingRect = None


    def drawSelectionRect(self):

        if self.separateDisplay:
            SCREEN = self.dualWindowSurf
        else:
            SCREEN = self.screen


        MODE_ON = self.mode == 1 or (self.separateDisplay and self.dualWindow.focused)
        
        if MODE_ON:
            if "r" in self.keypress and self.renderRect:
                self.renderRect = None
                self.clearCache()
                self.EXPORT = True
            if "mouse0" in self.keypress and not self.activeInputNode:
                self.selRectOrigin = self.mouse_pos.copy()

        if "mouse0" in self.keypress_held_down and self.selRectOrigin and MODE_ON:

            p = [min(self.selRectOrigin[0], self.mouse_pos[0]), min(self.selRectOrigin[1], self.mouse_pos[1])]
            s = [abs(self.selRectOrigin[0] - self.mouse_pos[0]), abs(self.selRectOrigin[1] - self.mouse_pos[1])]
            self.selectionRect = pygame.Rect(p, s)
            pygame.draw.rect(SCREEN, [255,255,255], self.selectionRect, width= 1)

        else:

            if self.selRectOrigin and not "mouse0" in self.keypress_held_down:
                if MODE_ON:
                    self.setRectToRendering(self.selectionRect)


            self.selRectOrigin = None
            self.selectionRect = None


    def setRectToRendering(self, r):
        print(r)
        if r.width < 10 or r.height < 10:
            return

        p1 = v2(r.topleft)
        p2 = p1 + v2(r.size)
        print(p1, p2)
        self.renderRect = [self.viewportToPixel(p1), self.viewportToPixel(p2)]
        print(self.renderRect)
        self.clearCache()
        self.EXPORT = True

    def viewportToPixel(self, pos):
        MPIM = pos - self.IMBLITPOS
        MPRATIO = MPIM / self.SCALEFACTOR
        return MPRATIO + self.topLeft
    
    def pixelToViewport(self, pixel_pos):
        MPRATIO = pixel_pos - self.topLeft
        MPIM = MPRATIO * self.SCALEFACTOR
        pos = MPIM + self.IMBLITPOS
        return pos
    
    def quickText(self, text, pos):
        t = self.fontRoboto.render(text, True, [255,0,0])
        r = t.get_rect()
        r.topleft = pos
        pygame.draw.rect(self.screen, [0,0,0], r)
        self.screen.blit(t, pos)



    def get_bounding_box(self, labeled_image, label):
        """
        Get the bounding box (min_x, min_y, width, height) of a specific particle label.
        
        :param labeled_image: The labeled image where each particle has a unique label.
        :param label: The specific label whose bounding box we want to find.
        :return: (min_x, min_y, width, height) representing the bounding box.
        """
        # Find all the positions where the label exists
        positions = np.where(labeled_image == label)

        if len(positions[0]) == 0:  # Label not found
            return None

        # Get the min and max coordinates
        min_x, max_x = positions[0].min(), positions[0].max()
        min_y, max_y = positions[1].min(), positions[1].max()

        return min_x, min_y, max_x, max_y
    
    def tickContextMenus(self):
        FAR = True
        for x in self.contextMenuShow:
            x.show()
            if x.FAR == False:
                FAR = False

        if FAR:
            self.contextMenuShow.clear()

    def updateNodes(self):

        self.MANIPULATENODES = True
        if self.separateDisplay:
            self.MANIPULATENODES = not self.dualWindow.focused
        
        if "esc" in self.keypress and self.CURR_NODES != self.NODES:
            self.CURR_NODES = self.NODES
            self.pipelineImage = None
            #self.clearCache()
            self.EXPORT = True

        for x in self.CURR_NODES:
            x.render()


        if self.activeInputNode:

            self.activeInputNode.cancelKH.active = True
            draw_bezier_curve(self.screen, [255,0,0], self.activeInputNode.pos + self.activeInputNode.parent.pos, self.mouse_pos, steps = 15, width = 2)
            if "mouse2" in self.keypress:
                self.activeInputNode = None

        if self.activeNode and "mouse2" in self.keypress and "shift" not in self.keypress_held_down:
            self.DELTA = self.mouse_pos - self.activeNode.pos
            self.clicked = self.activeNode

        if self.clicked and "mouse2" not in self.keypress_held_down:

            if self.rightClickTick < 10:
                print("OPEN CONTEXT MENU")

                self.contextMenuShow.clear()
                for x in self.clicked.contextMenus:
                    x.toggleOn(self.mouse_pos)


            if self.connectionUnderMouse:
                NODERIGHT = self.connectionUnderMouse
                NODELEFT = self.connectionUnderMouse.IN_CONNECTION

                if self.clicked.inputs:
                    INPUT = None
                    for x in self.clicked.inputs:
                        if not x.selfInput:
                            INPUT = x
                            break
                    if INPUT:
                        NODELEFT.disconnect()
                        INPUT.disconnect()
                        NODELEFT.connect(INPUT)

                if self.clicked.outputs:
                    OUTPUT = self.clicked.outputs[0]
                    NODERIGHT.disconnect()
                    OUTPUT.disconnect()

                    OUTPUT.connect(NODERIGHT)

                self.EXPORT = True

            for x in self.clicked.inputs + self.clicked.outputs:
                if x.connectOnRelease:
                    x.connect(x.connectOnRelease)

            self.clicked = None


        if self.clicked and "mouse2" in self.keypress_held_down:
            self.clicked.realPos = self.mouse_pos.copy() - self.DELTA



    def execCalc(self):
        if self.EXPORT:
            self.EXPORT = False
            godNodes = []
            for x in self.CURR_NODES:
                x.calcedThisExport = False
                if x.godNode and not x.disabled:
                    godNodes.append(x)
            if not self.CALCING:  
                self.CALCING = True
                start_new_thread(self.multipleGodNodeCalc, (godNodes, ))




    def multipleGodNodeCalc(self, nodes):
        self.CALCING = True
        for x in nodes:
            x.calc()
        self.CALCING = False


    def startProcessing(self, labels):
        if self.PROCESSING:
            return
        self.PROCESSING = True
        start_new_thread(self.processData, (labels, ))


    def processData(self, labels):


        try:

            """
            Precomputes all necessary particle data (bounding boxes, areas, etc.)
            and stores them in a dictionary for fast retrieval.
            """
            self.PROCESSEDDATA = {}  # Dictionary to store precomputed data
            self.PROCESSEDDATA["labelArray"] = labels  # Store original labeled image
            self.PROCESSEDDATA["bounding_boxes"] = {}  # Store bounding boxes
            self.PROCESSEDDATA["areas"] = {}  # Store particle areas
            self.PROCESSEDDATA["pixel_counts"] = {}  # Store pixel count per particle
            self.PROCESSEDDATA["max_diameter"] = {}
            self.PROCESSEDDATA["max_diameter_real"] = {}
            self.PROCESSEDDATA["perimeter"] = {}
            self.PROCESSEDDATA["dispersion"] = 0



            pixel_area, unit = self.FOV.getAreaOfPixel()  # Get area of a single pixel

            pixel_length = self.FOV.getPixelToDistance()

            labels_2d = labels[..., 0]
            
            self.PROCESSEDDATA["unit"] = unit

            unique_labels = np.unique(labels)  # Get all unique labels in the image

            areaOfImage = self.FOV.resolution[0] * self.FOV.resolution[1]
            self.PROCESSEDDATA["areaOfImage"] = areaOfImage * pixel_area

            self.PROCESSEDDATA["perimeterByArea"] = 0
            totalPerimeter = 0

            for label in unique_labels:
                self.processingProgress = f"{label}/{len(unique_labels)}"
                if label == 0:
                    continue  # Skip background


                # Create a binary mask for the current label
                binary_mask = (labels_2d == label)

                # Compute the perimeter of the current label
                label_perimeter = perimeter(binary_mask, 8)  # 8-connectivity for better accuracy
                self.PROCESSEDDATA["perimeter"][label] = label_perimeter

                # Add to total perimeter
                totalPerimeter += label_perimeter


                # Compute bounding box
                min_x, min_y, max_x, max_y = self.get_bounding_box(labels, label)
                self.PROCESSEDDATA["bounding_boxes"][label] = (min_x, min_y, max_x + 1, max_y + 1)

                # Count number of pixels belonging to this label
                particle_pixels = np.sum(labels_2d == label)
                self.PROCESSEDDATA["pixel_counts"][label] = particle_pixels

                # Compute real-world area
                real_area = particle_pixels * pixel_area
                self.PROCESSEDDATA["areas"][label] = real_area
                try:
                    maxD = fast_mean_feret_diameter(labels_2d, label)
                except:
                    traceback.print_exc()
                    maxD = 0
                self.PROCESSEDDATA["max_diameter"][label] = maxD
                self.PROCESSEDDATA["max_diameter_real"][label] = maxD*pixel_length

            self.PROCESSEDDATA["perimeterByArea"] = (totalPerimeter * pixel_length) / areaOfImage


            self.PROCESSING = False  # Mark processing as done
            self.processingProgress = ""
            self.getDispersion()
            print("Processing complete! Precomputed data is ready.")
        except:
            traceback.print_exc()
            print("Processing halted due to error.")
            self.PROCESSING = False



    def viewData(self):

        if self.PROCESSING:
            return
        
        if not self.PROCESSEDDATA:
            return
        
        if self.THRESHOLD.render():
            self.getDispersion()
        if self.PROCESSEDDATA:
            if "dispersion" in self.PROCESSEDDATA:
                DISPERSION = self.PROCESSEDDATA["dispersion"]
                self.debugText(f"DISPERSION: {DISPERSION:.1f}")

        try:
            pixel = self.viewportToPixel(self.mouse_pos)
            labelUnderMouse = self.PROCESSEDDATA["labelArray"][int(pixel[0]), int(pixel[1])]
            labelUnderMouse = int(labelUnderMouse[0])  # Convert first element to an integer

            perimeterPerArea = self.PROCESSEDDATA["perimeterByArea"]
            self.debugText(f"Perimeter/Area {perimeterPerArea}")

            if labelUnderMouse == 0:  # Skip background (label 0)
                return
            
            # Retrieve precomputed bounding box
            min_x, min_y, max_x, max_y = self.PROCESSEDDATA["bounding_boxes"][labelUnderMouse]
            origin = self.pixelToViewport([min_x, min_y])
            endpos = self.pixelToViewport([max_x, max_y])

            # Draw bounding box
            r2 = pygame.Rect(origin, [endpos[0] - origin[0], endpos[1] - origin[1]])
            pygame.draw.rect(self.screen, [255, 0, 0], r2, width=1)

            # Retrieve precomputed particle area
            particle_pixels = self.PROCESSEDDATA["pixel_counts"][labelUnderMouse]
            particleArea = self.PROCESSEDDATA["areas"][labelUnderMouse]

            maxD = self.PROCESSEDDATA["max_diameter"][labelUnderMouse]
            pixelDistance = self.FOV.getPixelToDistance() 

            perimeter = self.PROCESSEDDATA["perimeter"][labelUnderMouse]

            # Display text
            y = 20
            for text in [f"Pixel area: {particle_pixels} px", f"Particle area: {particleArea:.1f} {self.FOV.unit}²", 
                         f"Maximum Diameter: {maxD * pixelDistance:.1f} {self.FOV.unit} ({maxD:.0f}px)",
                         f"Perimeter: {perimeter * pixelDistance:.1f} {self.FOV.unit} ({perimeter:.0f}px)"]:
                self.quickText(text, self.mouse_pos + [20, y])
                y += 20

            
            


            

        except:
            pass

       
            

    def getDispersion(self):
        threshold = self.THRESHOLD.value  # Get the threshold diameter

        if not self.PROCESSEDDATA["areas"]:
            return

        # Sum the areas of particles whose diameter exceeds the threshold
        self.total_undispersed_area = sum(
            area for label, area in self.PROCESSEDDATA["areas"].items()
            if self.PROCESSEDDATA["max_diameter_real"][label] > threshold
            )
        print(self.total_undispersed_area, self.PROCESSEDDATA["areaOfImage"])
        print("URF:", self.total_undispersed_area / self.PROCESSEDDATA["areaOfImage"])
        self.PROCESSEDDATA["dispersion"] = 100 - (self.total_undispersed_area/self.PROCESSEDDATA["areaOfImage"])*100
        print("Ned dispersion:", self.PROCESSEDDATA["dispersion"])

    def debugText(self, text):

        t = self.fontSmaller.render(str(text), True, [255,255,255])
        self.screen.blit(t, [self.res[0] - 20 - t.get_size()[0], 200 + self.debugI * 12])
        self.debugI += 1
        
    def handleMouseImPos(self):

        if self.separateDisplay:

            if not self.dualWindow.focused:
                return

            SCREEN = self.dualWindowSurf
        else:
            SCREEN = self.screen

        t = self.fontSmall.render(f"{self.mousePosIm[0]:.0f}, {self.mousePosIm[1]:.0f}", True, [150,150,150])

        POS = self.mouse_pos - t.get_rect().center - [0, 20]

        POS[0] = max(POS[0], 5)
        POS[1] = max(POS[1], 5)

        SCREEN.blit(t, POS)


        

    def displayIm(self):

        #self.zoom += 0.01

        if self.separateDisplay:
            SCREEN = self.dualWindowSurf
            RES = v2(SCREEN.size)
        else:
            SCREEN = self.screen
            RES = self.res


        MANIPULATE = self.mode == 1 or (self.separateDisplay and self.dualWindow.focused)

        if MANIPULATE:
            if "wheelUp" in self.keypress:
                self.zoom += 1
            elif "wheelDown" in self.keypress:
                self.zoom -= 1  

        

        self.imageSize = v2(self.imageApplied.get_size())
        
        self.zoom = np.clip(self.zoom, -2, 15) 

        self.zoomDelta = 0.5 * self.zoomDelta + 0.5 * self.zoom

        zoom_factor = (1*(16-self.zoomDelta)/15)**1.5

        if "mouse1" in self.keypress_held_down and MANIPULATE:
            
            self.imagePos += self.mouseDelta

        new_size = self.imageSize * zoom_factor

        CANVASRATIO = RES[0] / RES[1]
        IMAGERATIO = self.imageSize[0] / self.imageSize[1]
        mod_zoom_dim = self.imageSize.copy()
        cropped_inside = 0

        if CANVASRATIO > IMAGERATIO:
            mod_zoom_dim[0] = self.imageSize[1] * CANVASRATIO
            if new_size[1] * CANVASRATIO > self.imageSize[0]:
                new_size[0] = self.imageSize[0]
            else:
                cropped_inside = 1
                new_size[0] = new_size[1]*CANVASRATIO
        else:
            mod_zoom_dim[1] = self.imageSize[0] / CANVASRATIO
            if new_size[0]/CANVASRATIO > self.imageSize[1]:
                new_size[1] = self.imageSize[1]
            else:
                cropped_inside = 1
                new_size[1] = new_size[0]/CANVASRATIO

        xDim, yDim = new_size


        self.crop_rect = pygame.Rect(self.imageSize/2 - self.imagePos, [0,0])

        

        self.crop_rect.inflate_ip(xDim, yDim)

        self.crop_rect.x = max(0, self.crop_rect.x)
        self.crop_rect.y = max(0, self.crop_rect.y)

        self.crop_rect.x = min(self.crop_rect.x, self.imageSize[0] - xDim)
        self.crop_rect.y = min(self.crop_rect.y, self.imageSize[1] - yDim)

        self.imagePos = self.imageSize/2 - self.crop_rect.center

        


        TEMPIM = self.imageApplied.subsurface(self.crop_rect).copy()

       


        self.SCALEFACTOR = min(RES[0] / self.crop_rect.size[0], RES[1] / self.crop_rect.size[1])

        TEMPIM = pygame.transform.scale_by(TEMPIM, self.SCALEFACTOR)

        self.IMBLITPOS = RES/2 - v2(TEMPIM.get_size())/2
        SCREEN.blit(TEMPIM, self.IMBLITPOS)

        MPIM = self.mouse_pos - self.IMBLITPOS

        #MPIM = [MPIM[0] / self.res[0], MPIM[1] / self.res[1]]

        MPRATIO = MPIM / self.SCALEFACTOR   #v2(MPIM[0] * crop_rect.width/new_size[0], MPIM[1] * crop_rect.height/new_size[1])

        self.topLeft = v2(self.crop_rect.x, self.crop_rect.y)
        
        self.mousePosIm = self.viewportToPixel(self.mouse_pos)
 
        self.debugText(str(self.crop_rect))


    def pos2screen(self, pos):
        print(pos)

    def texts(self):
        

        self.screen.blit(self.VIGNETTE, [self.res[0]/2 - self.VIGNETTE.get_width()/2, 0])

        for i, x in enumerate(["NODES", "VIEWPORT", "PIPELINE"]):

            diff = min(abs(i - self.modeX), 0.25)

            color = 255 * (1 - diff)
            
            tR = self.font.render(x, True, [color,color,color])

            MIDPOS = [self.res[0]/2, 25] - v2(tR.get_size()) / 2
            xPos = (i - self.modeX) * 250

            MIDPOS[0] += xPos
            self.screen.blit(tR, MIDPOS)

            r = tR.get_rect()
            r.topleft = MIDPOS
            if self.MANIPULATENODES and r.collidepoint(self.mouse_pos) and "mouse0" in self.keypress:
                self.mouseAvailable = True
                self.mode = i



        t2 = "Press SPACE to cycle modes"

        tR = self.fontSmaller.render(t2, True, [255,255,255])
        self.screen.blit(tR, [self.res[0]/2, 45] - v2(tR.get_size()) / 2)

        if self.pipelineImage:
            self.debugText(f"Pipeline image: {self.pipelineImage}")

        elif self.imPath:
            self.debugText(self.imPath)

        if self.modelLoaded:
            self.debugText(f"Staged model: {self.modelLoaded}")
        
        self.debugText(f"FPS: {self.clock.get_fps():.0f}")
        self.debugText(f"IDLE: {self.dT*100:.0f}%")
        

        if self.CALCING or self.PROCESSING:
            strs = ["", ".", "..", "..."]
            if self.calcTick.tick():
                self.strI += 1
                if self.strI == len(strs):
                    self.strI = 0

            dotStr = strs[self.strI]
            if self.CALCING:
                self.debugText(f"Generating{dotStr}")
            if self.PROCESSING:
                self.debugText(f"Processing particles: {self.processingProgress}")

        

    def initLoop(self):

        self.initScreen()

        while 1:
            self.debugI = 0

            t = time.time()
            self.loop()
            dT = time.time() - t
            
            self.texts()

            pygame.display.update()
            if self.separateDisplay:
                self.dualWindow.flip()
            self.clock.tick(60)

            tT = time.time() - t

            self.dT = (1 - dT/tT) * 0.1 + self.dT * 0.9


def getUntrainedImages():

    print("Loading image...")


    MAINPATH = os.getcwd()

    print(MAINPATH)
    
    trainImsPath = os.path.join(MAINPATH, "AI/train_images/")
    trainIms = os.listdir(trainImsPath)
    print(trainIms)
    trainImNames = " ".join(trainIms)
    print(trainImNames)
    

    IMAGES = []
    PATH = os.path.join(MAINPATH, "misc/kaikki/")
    for x in os.listdir(PATH):
        IMAGES.append(PATH + x)

    PATH = os.path.join(MAINPATH, "misc/kaikki2/")
    for x in os.listdir(PATH):
        IMAGES.append(PATH + x)


    imagesTrained = []
    imagesUntrained = []
    for x in IMAGES:
        imName = Path(x).stem
        if imName in trainImNames:
            print("IMAGE ALREADY PRESENT")
            imagesTrained.append(imName)
        else:
            imagesUntrained.append(x)
    return imagesUntrained

if __name__ == "__main__":

    #getUntrainedImages()

    APP = App()