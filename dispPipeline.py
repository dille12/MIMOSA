from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import App
import scipy.ndimage as sc
import csv
from tkinter import filedialog, messagebox, simpledialog
import os
import core.nodeFuncs as NF
import tifffile, pygame
import numpy as np
import traceback
from _thread import start_new_thread
from skimage import measure
import time
from scipy.spatial.distance import pdist
from numba import jit
import math
import pyperclip
import core.funcs.toolTip as toolTip
from core.contextMenu import ContextMenu
from core.funcs.saveload import saveSetupClean, loadSetupClean
from scipy.stats import mode
import tkinter as tk
from core.globalArray import update_global_dispersion_data
def export_csv_with_dialog(csv_string, default_filename="dispersion_data.csv"):

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")],
        initialfile=default_filename,
        title="Export CSV"
    )

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_string)
        print(f"Exported to {file_path}")
    else:
        print("Export cancelled.")

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

def fetchData(txtfile) -> dict:

    with open(txtfile, encoding="utf-8", errors="ignore") as f:
        data = f.read()

    d = data.split("$")
    d2 = {}
    for x in d:
        x = x.rstrip("\n")
        y = x.split(" ")
        d2[y[0]] = y[1:]
    return d2

def loadImage(image_path):
    try:
        image_array = tifffile.imread(image_path)
        print("Loading image data...")
        with tifffile.TiffFile(image_path) as tif:
            volume = tif.asarray()
            axes = tif.series[0].axes
            imagej_metadata = tif.imagej_metadata
    except:
        print("Not a tiff file")
        im = pygame.image.load(image_path)
        image_array = pygame.surfarray.array3d(im)  
        image_array = image_array.transpose([1, 0, 2])

    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.stack([image_array] * 3, axis=-1)

    if image_array.shape[2] not in [1, 3]:
        image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)

    surface = pygame.surfarray.make_surface(np.transpose(image_array, (1, 0, 2))).convert()
    return image_array, surface


def sortData(file_paths):
    file_pairs = {}

    for file_path in file_paths:
        folder, filename = os.path.split(file_path)  # Get folder and filename
        name, ext = os.path.splitext(filename)  # Split filename and extension

        # Add to dictionary, grouping by filename (without extension)
        if name not in file_pairs:
            file_pairs[name] = {}  # Initialize entry

        if ext.lower() == ".txt":
            file_pairs[name]["txt"] = file_path
        elif ext.lower() == ".tif":
            file_pairs[name]["tif"] = file_path

    return file_pairs


def nice_number(value, round_to_nearest=True):
    exponent = math.floor(math.log10(value))
    fraction = value / 10**exponent

    if round_to_nearest:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10

    return nice_fraction * 10**exponent


from scipy.spatial import ConvexHull
@jit(nopython=True)
def compute_feret_diameters(hull_points, angles):
    feret_diameters = np.empty(len(angles), dtype=np.float32)
    
    for i in range(len(angles)):
        angle = angles[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        min_proj = 1e10
        max_proj = -1e10

        for j in range(hull_points.shape[0]):
            x, y = hull_points[j]
            proj = x * cos_a + y * sin_a
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)

        feret_diameters[i] = max_proj - min_proj

    return feret_diameters

def fast_mean_feret_diameter(binary_mask, num_angles=180):
    coords = np.column_stack(np.nonzero(binary_mask))

    if coords.shape[0] < 3:
        return 0.0

    # Use convex hull to simplify
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices].astype(np.float32)

    # Precompute angles
    angles = np.linspace(0, np.pi, num_angles, endpoint=False).astype(np.float32)

    # Compute Feret diameters
    feret_diameters = compute_feret_diameters(hull_points, angles)
    return float(np.mean(feret_diameters))

import skimage




class Particle:
    def __init__(self, image: "ImageDataObj", area, diameter, coords = None):
        image.PARTICLES.append(self)
        self.particle_pixels = area
        self.diameter = diameter
        self.coords = coords

        self.real_area = self.particle_pixels*image.pixel_area
        self.real_diameter = self.diameter * image.pixel_length



@jit(nopython=True)
def count_pixels_numba(label_array):
    max_label = label_array.max()
    counts = np.zeros(max_label + 1, dtype=np.int32)
    for y in range(label_array.shape[0]):
        for x in range(label_array.shape[1]):
            label = label_array[y, x]
            if label > 0:
                counts[label] += 1
    return counts


class ImageDataObj:
    def __init__(self, app: "App", key, d):
        self.tifPath = d["tif"]
        self.txtPath = d["txt"]
        self.app = app
        self.key = key

        self.resultarray = None
        self.resultsurface = None
        self.resultSmallSurface = None
        self.resultInspectSurface = None
        self.useInDispersionAnalysis = "undefined"
        self.LOADED = True
        self.deltaPos = None

        self.PARTICLES = []

        self.toolTip = toolTip.toolTip(self.app, self, self.key, "An image with loaded metadata. Drag with primary mouse button into a defined sample to include it in the dispersion analysis Press DEL to delete. If particle map present, right click to inspect. Note: images under UNSORTED are not processed.")
        try:
            self.image_array, self.surface = loadImage(self.tifPath)
        except:
            self.app.notify(f"Error loading image {self.tifPath}. Please check the file format or path.")
            traceback.print_exc()
            self.LOADED = False
            return

        self.INDIVIDUAL_NODES = []


        self.smallSurface = pygame.transform.scale_by(self.surface, 150/self.surface.get_size()[1])
        self.InspectSurface = pygame.transform.scale_by(self.surface, self.app.res[1]/self.surface.get_size()[1])


        ContextMenu(self, "Inspect", function=lambda: self.inspectSingleImage())
        ContextMenu(self, "Load to viewport", function=lambda: self.loadToViewport())
        ContextMenu(self, "Recalculate", function=lambda: self.reCalcThisImage())
        ContextMenu(self, "Make/View individual node structure", function=lambda: self.viewIndividual())
        ContextMenu(self, "Delete individual node structure", function=lambda: self.delIndividual())

        d = fetchData(self.txtPath)
        self.FOVVal = d["CM_FIELD_OF_VIEW"]
        self.IMAGERES = d["CM_IMAGE_SIZE"]
        self.MAGNIFICATION = d["CM_MAG"][0]
        self.SCANROTATION = float(d["SM_SCAN_ROTATION"][0])
        self.STAGEPOSITION = d["CM_STAGE_POSITION"][0:2]
        self.STAGEPOSITION = [float(x) for x in self.STAGEPOSITION]
        print("Stage Position:", self.STAGEPOSITION)
        res = [int(x) for x in self.IMAGERES]
        print("RES:", res)
        self.FOV = NF.parseToFOV(self.FOVVal, res)
        print(self.FOV)
        if self.FOV:
            self.parsed = True
        else:
            self.parsed = False

        self.t = self.app.fontSmall.render(self.key, True, [255,255,255])
        self.tr = self.app.fontSmall.render(self.key, True, [255,0,0])
        app.SAMPLEDATA["Unsorted"].images.append(self)


    def delIndividual(self):
        if self.INDIVIDUAL_NODES:
            self.INDIVIDUAL_NODES.clear()
            self.app.notify("Individual node structure deleted")

            if self.app.CURR_NODES == self.INDIVIDUAL_NODES:
                self.app.CURR_NODES = self.app.NODES


        else:
            self.app.notify("No individual node structure present")

    def viewIndividual(self):
        if not self.INDIVIDUAL_NODES:
            serialized = saveSetupClean(self.app, returnSer=True)
            loadSetupClean(self.app, self.INDIVIDUAL_NODES, serialized = serialized)

        self.app.CURR_NODES = self.INDIVIDUAL_NODES
        self.app.clearCache()
        self.app.mode = 0
        self.app.setBackground(self.surface)
        self.app.pipelineImage = self.surface
        self.EXPORT = True

    def loadToViewport(self):
        self.app.loadImage(self.tifPath)
        self.app.clearCache()
        self.app.mode = 0
        self.EXPORT = True

        

    def addResult(self, image):
        self.resultarray = image
        self.resultsurface = pygame.surfarray.make_surface(image)
        self.resultSmallSurface = pygame.transform.scale_by(self.resultsurface, 150/self.resultsurface.get_size()[1])
        self.resultInspectSurface = pygame.transform.scale_by(self.resultsurface, self.app.res[1]/self.resultsurface.get_size()[1])
        self.useInDispersionAnalysis = "undefined"


    def batch_particle_max_diameters_fast(self, labeled_image):         
        """         
        Faster version using convex hull for each particle         
        """         
        props = skimage.measure.regionprops(labeled_image)         
        max_diameters = {}
        areas = {}
        coordinates = {}
        self.app.PIPELINESUBPROGRESS = 0         
        for i, prop in enumerate(props):              
            self.app.PIPELINESUBPROGRESS = i/len(props)              
            coords = prop.coords
            areas[prop.label] = prop.area
            coordinates[prop.label] = coords
                            
            if len(coords) < 2:                 
                max_diameters[prop.label] = 1.0
                print("Single pixel particle found")          
            elif len(coords) < 4:                 
                # Too few points for convex hull                 
                distances = pdist(coords)                 
                max_diameters[prop.label] = distances.max()             
            else:                 
                try:                     
                    # Use convex hull to reduce computation                     
                    hull = ConvexHull(coords)                     
                    hull_points = coords[hull.vertices]    
                    coordinates[prop.label] = hull_points                 
                    distances = pdist(hull_points)                     
                    max_diameters[prop.label] = distances.max()                 
                except:                     
                    # Fallback to all points if convex hull fails                     
                    distances = pdist(coords)                     
                    max_diameters[prop.label] = distances.max()                  
        
        return max_diameters, areas, coordinates

    def getParticles(self):
        self.tempProgressImage = self.resultarray.copy()
        self.tempProgressImage[:, :, 1:3] = 0

        self.app.setBackground(pygame.surfarray.make_surface(self.tempProgressImage))
        im2d = self.resultarray[:, :, 0]
        labels = measure.label(im2d)# Label connected components
        self.labelArray = labels  
        self.pixel_length = self.FOV.getPixelToDistance()  # Convert pixels to real-world distance
        
        print("PIXEL LENGTH:", self.pixel_length)
        self.pixel_area = self.FOV.getAreaOfPixel()[0]

        self.image_area = self.pixel_area * self.image_array.shape[0] * self.image_array.shape[1]

        print("PIXEL AREA:", self.pixel_area)
        print("IMAGE AREA:", self.image_area)
        print("OLD IMAGE AREA:", self.FOV.getAreaOfImage())

        max_diameters, areas, coordinates = self.batch_particle_max_diameters_fast(labels)

        self.PARTICLES = []

        for label in areas:
            d = max_diameters[label]
            a = areas[label]
            c = coordinates[label]
            if d > 100:
                print("AREA:", a, "DIAMETER:", d, "COORDS:", c)
            Particle(self, a, d, coords = c)

        self.app.AiProcessRect = None


    def inspectSingleImage(self):

        if isinstance(self.resultarray, np.ndarray):
            self.app.InspectImages = [self]
            self.app.pipelineInspect = True
            self.app.inspectToggle = True
            
        else:
            self.app.notify("No result image present for inspection")


    def reCalcThisImage(self):
        if self.app.PIPELINEACTIVE:
            self.app.notify("Already running an operation!")
        else:
            start_new_thread(PIPELINEWRAP, (self.app, [self]))


    def tick(self, sample):
        i = sample.images.index(self)
        ypos = i*30

        POS = [sample.xpos, 280 + ypos]
        
        r = self.t.get_rect()
        r.topleft = POS
        r.height = 30

        if self.INDIVIDUAL_NODES:
            if self.app.CURR_NODES == self.INDIVIDUAL_NODES:
                additional = "[C-VIEWING]"
            else:
                additional = "[C]"
        else:
            additional = ""


        blitTextStr = f"{self.key} {additional}"


        if self.app.PIPELINERESULT == self:
            pygame.draw.rect(self.app.screen, [0,255,255], r, width = 1)

        if (r.collidepoint(self.app.mouse_pos) or self.app.pickedUp == self) and not self.app.contextMenuShow:
            blittext = blittext = self.app.fontSmall.render(blitTextStr, True, self.app.MAINCOLOR)

            POS2 = self.app.res - self.smallSurface.get_size() - [10, 10]

            if self.resultSmallSurface:
                self.app.screen.blit(self.resultSmallSurface, POS2)
            else:
                self.app.screen.blit(self.smallSurface, POS2)

            

            tTemp = self.app.fontSmall.render(f"Image: {self.key}", True, [255,255,255])
            self.app.screen.blit(tTemp, POS2 - [tTemp.get_size()[0] + 10, 0])
            
            unit = "um"

            tTemp = self.app.fontSmall.render(f"FOV: {self.FOV.values[0]}{unit} x {self.FOV.values[1]}{unit}", True, [255,255,255])
            self.app.screen.blit(tTemp, POS2 - [tTemp.get_size()[0] + 10, -30])

            tTemp = self.app.fontSmall.render(f"MAG: {self.MAGNIFICATION}", True, [255,255,255])
            self.app.screen.blit(tTemp, POS2 - [tTemp.get_size()[0] + 10, -60])

            if "mouse0" in self.app.keypress:
                self.app.pickedUp = self

            elif "mouse2" in self.app.keypress:
                self.app.contextMenuShow.clear()
                for x in self.contextMenus:
                    x.toggleOn(r.topleft)
                
            elif "del" in self.app.keypress:
                sample.images.remove(self)
                if sample.text != "Unsorted":
                    sample.genImageMap()
                return
            
            self.toolTip.render()

        else:

            if isinstance(self.resultarray, np.ndarray):
                if self.useInDispersionAnalysis == "undefined":
                    blittext = self.app.fontSmall.render(blitTextStr, True, [255,255,100])
                elif self.useInDispersionAnalysis == "use":
                    blittext = self.app.fontSmall.render(blitTextStr, True, [100,255,100])
                else:
                    blittext = self.app.fontSmall.render(blitTextStr, True, [255,100,100])
            else:
                blittext = blittext = self.app.fontSmall.render(blitTextStr, True, [255,255,255])


        



        if self.app.pickedUp == self:
            HIGHLIGHTEDSAMPLE = None
            POS = self.app.mouse_pos + [20,20]
            for x in self.app.SAMPLEDATA:
                r = self.app.SAMPLEDATA[x].rect
                if r.collidepoint(self.app.mouse_pos):
                    pygame.draw.rect(self.app.screen, [255,255,255], r, width=1)
                    HIGHLIGHTEDSAMPLE = self.app.SAMPLEDATA[x]

            if "mouse0" not in self.app.keypress_held_down:
                self.app.pickedUp = None
                if HIGHLIGHTEDSAMPLE:
                    sample.images.remove(self)
                    HIGHLIGHTEDSAMPLE.images.append(self)
                    if HIGHLIGHTEDSAMPLE.text != "Unsorted":
                        HIGHLIGHTEDSAMPLE.genImageMap()
                    if sample.text != "Unsorted":
                        sample.genImageMap()

        if not self.deltaPos:
            self.deltaPos = self.app.v2(POS)

        self.deltaPos = self.app.v2(POS) * 0.25 + self.deltaPos * 0.75
        self.app.screen.blit(blittext, self.deltaPos)


        

class Sample:
    def __init__(self, app: "App", text, xpos):
        self.images = []
        self.app = app
        self.text = text
        self.t = app.font.render(text, True, [255,255,255])
        self.tr = app.font.render(text, True, self.app.MAINCOLOR)
        self.w = max(150, self.t.get_rect().width + 50)
        self.xpos = xpos
        self.rect = pygame.Rect(xpos, 250, self.w, 1000)
        self.URF = 0
        self.infoSurf = pygame.Surface((500,500)).convert()
        self.particleData = []
        self.animTick = 0
        self.dispAreaInfo = False
        self.additionaldata = {}

        ContextMenu(self, "View dispersion data", function=lambda: self.viewData())
        ContextMenu(self, "Rename", function=lambda: self.rename())
        ContextMenu(self, "Delete", function=lambda: self.delete())


        self.binAmount = 40

        self.genImageMap()


    def genImageMap(self):
        minx = float("inf")
        miny = float("inf")
        maxx = -float("inf")
        maxy = -float("inf")
        RECTS = []
        for x in self.images:
            stagePos = pygame.math.Vector2(x.STAGEPOSITION.copy())
            

            print("StagePos", stagePos)
            imsize = x.FOV.getResolution("mm")
            print("IMSIZE:", imsize)
            rect = [stagePos[0] - imsize[0]/2, stagePos[1] - imsize[1]/2, imsize[0], imsize[1]]
            minx = min(minx, rect[0])
            miny = min(miny, rect[1])
            maxx = max(maxx, rect[0] + rect[2])
            maxy = max(maxy, rect[1] + rect[3])
            print("Rect:", rect)
            RECTS.append([x, rect])

        print("MINX:", minx, "MINY:", miny, "MAXX:", maxx, "MAXY:", maxy)

        self.map = pygame.Surface([400,400]).convert()
        self.map.fill((50,0,0))
        SCALEX = maxx - minx
        SCALEY = maxy - miny
        SCALE = max(SCALEX, SCALEY)
        RECTS.sort(key=lambda rect: rect[1][3])  # Sort by the fourth value in the rect tuple
        RECTS.reverse()
        for image, x in RECTS:
            x[0] = (x[0] - minx) / SCALE * 400
            x[1] = (x[1] - miny) / SCALE * 400

            x[2] = x[2] / SCALE * 400
            x[3] = x[3] / SCALE * 400

            print("Drawing rect:", x)

            im = pygame.transform.scale(image.surface, [int(x[2]), int(x[3])])
            self.map.blit(im, [x[0], x[1]])

            pygame.draw.rect(self.map, [255,255,255], x, width=1)


        print("Map created")


    def additionaldata_to_csv_row(self, keys=None):
        if keys is None:
            keys = list(self.additionaldata.keys())
        values = [self.text] + [self.additionaldata.get(k, "") for k in keys]
        return ",".join(f'"{v}"' for v in values)
    
    def additionaldata_csv_header(self, keys=None):
        if keys is None:
            keys = list(self.additionaldata.keys())
        return ",".join(['"Sample Name"'] + [f'"{k}"' for k in keys])


    def compute_urf99(self, particles, total_area):
        sorted_particles = sorted([p for p in particles if p.real_diameter > 0], 
                                key=lambda x: x.real_diameter, reverse=True)
        cumulative_area = 0.0
        target_area = 0.01 * total_area

        for p in sorted_particles:
            cumulative_area += p.real_area
            if cumulative_area >= target_area:
                return p.real_diameter  # URF99 diameter threshold

        return 0.0  # fallback: all particles included



    def compileDispersionData(self):
        PARTICLES = []
        self.particleData = []
        self.unDispersedArea = 0
        self.totalArea = 0

        for image in self.images:
            if image.useInDispersionAnalysis == "use":
                PARTICLES += image.PARTICLES
                self.totalArea += image.image_area

        undispersed = 0
        self.minSize = float("inf")
        self.maxSize = 0

        self.areas = []
        diameters = []

        for particle in PARTICLES:
            d = particle.real_diameter
            if d >= 5:
                self.unDispersedArea += particle.real_area
                undispersed += 1
            self.minSize = min(d, self.minSize)
            self.maxSize = max(d, self.maxSize)
            self.areas.append(particle.real_area)
            if d > 0:
                diameters.append(d)

        self.averageArea = np.mean(self.areas)
        self.medianArea = np.median(self.areas)

        # Convert to numpy array
        diameters = np.array(diameters)
        d_sorted = np.sort(diameters)

        # Percentiles
        d10 = np.percentile(d_sorted, 10)
        d50 = np.percentile(d_sorted, 50)
        d90 = np.percentile(d_sorted, 90)
        span = (d90 - d10) / d50 if d50 > 0 else np.nan
        pdi = span

        # Central tendency and spread
        mean_d = np.mean(d_sorted)
        median_d = np.median(d_sorted)
        std_d = np.std(d_sorted)

        # Mode
        mode_result = mode(d_sorted, keepdims=False)
        mode_d = mode_result.mode if mode_result.count > 1 else np.nan

        # D[3,2] and D[4,3]
        d32 = np.sum(d_sorted**3) / np.sum(d_sorted**2) if np.sum(d_sorted**2) > 0 else np.nan
        d43 = np.sum(d_sorted**4) / np.sum(d_sorted**3) if np.sum(d_sorted**3) > 0 else np.nan

        print("TOTAL PARTICLES:", len(PARTICLES), "UNDISPERSED:", undispersed)
        print("RANGE:", self.minSize, self.maxSize)

        self.PARTICLES = PARTICLES
        self.UNDISPERSEDPARTICLES = undispersed

        self.fillerarea = np.sum(x.real_area for x in PARTICLES if x.real_diameter >= 0)
        self.undispfillerarea = np.sum(x.real_area for x in PARTICLES if x.real_diameter >= 5)

        self.URF = self.undispfillerarea / self.totalArea
        self.particleData = create_log_histogram(diameters, bins=self.binAmount)
        self.particleDataArea = create_log_area_histogram(PARTICLES, bins=self.binAmount)
        self.URF99 = self.compute_urf99(PARTICLES, self.totalArea)

        flat = "|".join(f"{c},{b}" for c, b in zip(self.particleData['bin_centers'], self.particleData['counts']))
        self.ExportData = flat

        print("URF:", self.URF, "FILLER AREA:", self.fillerarea, "UNDISP FILLER AREA:", self.undispfillerarea)

        self.additionaldata = {
            "URF": self.URF,
            "FILLER AREA": f"{self.fillerarea:.3f}um^2",
            "UNDISP FILLER AREA": f"{self.undispfillerarea:.3f}um^2",
            "MIN DIAMETER": f"{self.minSize:.3f}um",
            "MAX DIAMETER": f"{self.maxSize:.3f}um",
            "MEAN DIAMETER": f"{mean_d:.3f}um",
            "MEDIAN DIAMETER": f"{median_d:.3f}um",
            "MODE DIAMETER": f"{mode_d:.3f}um" if not np.isnan(mode_d) else "N/A",
            "STANDARD DEVIATION": f"{std_d:.3f}um",
            "D10": f"{d10:.3f}um",
            "D50": f"{d50:.3f}um",
            "D90": f"{d90:.3f}um",
            "SPAN": f"{span:.3f}",
            "SAUTER MEAN DIAMETER": f"{d32:.3f}um",
            "VOLUME-WEIGHTED MEAN DIAMETER": f"{d43:.3f}um",
            "AVERAGE AREA": f"{self.averageArea:.3f}um^2",
            "MEDIAN AREA": f"{self.medianArea:.3f}um^2",
            "UNDISPERSED PARTICLES": self.UNDISPERSEDPARTICLES,
            "TOTAL PARTICLES": len(PARTICLES),
            "TOTAL AREA": f"{self.totalArea:.3f}um^2",
            "URF99": f"{self.URF99:.3f}um",
            "EXPORT": f"{self.ExportData}"
        }
        try:
            update_global_dispersion_data(self.text, self.additionaldata, app = self.app)
        except Exception as e:
            print("Error updating global dispersion data:", e)
            traceback.print_exc()




    def drawInfo(self):

        minParticleSize = float("inf")
        for x in self.images:
            minParticleSize = min(x.FOV.getPixelToDistance() * 1, minParticleSize)

        self.infoSurf.fill((0,0,0))
        maxX = 0
        maxY = 0

        POS2 = self.app.res - self.map.get_size() - [10, 10]

        for i, text in enumerate([
            f"URF : {self.URF}",
            f"Minimum Particle Size: {minParticleSize:.3f}um"
        ]):
            t = self.app.fontSmaller.render(text, True, [255,255,255])
            self.infoSurf.blit(t, [0, i*15])
            maxX = max(maxX, t.get_size()[0])
            maxY = max(maxY, i*15)
        
        self.app.screen.blit(self.infoSurf, POS2 - [0,40 + maxY], area=[0,0,maxX+20, maxY+20])

        self.app.screen.blit(self.map, POS2)


    def delete(self):
        self.app.SAMPLEDATA["Unsorted"].images += self.images
        del self.app.SAMPLEDATA[self.text]


    def rename(self):
        text = simpledialog.askstring("Input", "Enter new name of the sample")
        if not text:
            return

        else:
            del self.app.SAMPLEDATA[self.text]
            self.text = text
            self.t = self.app.font.render(self.text, True, [255,255,255])
            self.tr = self.app.font.render(self.text, True, self.app.MAINCOLOR)
            self.app.SAMPLEDATA[self.text] = self


    
    def displayInfo(self):
        DATA = self.particleData if not self.dispAreaInfo else self.particleDataArea

        bin_x = 300
        maxWidth = 1000
        
        binWidth = maxWidth/self.binAmount
        self.animTick = self.animTick * 0.88 + 0.12


        x_sampleTitles = bin_x
        for x in self.app.SAMPLEDATA:

            title = self.app.SAMPLEDATA[x].text

            if title == "Unsorted":
                continue
            if title == self.text:
                c = [255,255,255]
            else:
                c = [100,100,100]


            t = self.app.font.render(title, True, c)

            r = t.get_rect()

            POS = [x_sampleTitles, 200 - t.get_size()[1]]
            r.topleft = POS

            if r.collidepoint(self.app.mouse_pos):
                t = self.app.font.render(title, True, [255,0,0])
                if "mouse0" in self.app.keypress:
                    self.app.SAMPLEDISPLAY = self.app.SAMPLEDATA[x]
                    self.animTick = 0


            self.app.screen.blit(t, POS)
            x_sampleTitles += t.get_size()[0] + 30
            
            


        
        particleMax = np.max(DATA["counts"])
        r = pygame.Rect(bin_x, 500 - 200, self.binAmount*binWidth, 200)
        r.inflate_ip(4,4)

        t = self.app.fontSmall.render("Particle Size Distribution", True, [255,255,255])
        self.app.screen.blit(t, [bin_x, 500 - 200 - t.get_size()[1] - 10])

        pygame.draw.rect(self.app.screen, [255,255,255], r, width=1)

        underMouse = None

        desired_major_ticks = 10
        rough_tick_interval = particleMax / desired_major_ticks

        tick_interval = nice_number(rough_tick_interval)   


        for i in range(int(int(particleMax)//tick_interval)):
            i = i + 1
            i2 = i*tick_interval/particleMax * 200
            y = 500 - i2
            pygame.draw.line(self.app.screen, [50,50,50], [bin_x, y], [bin_x + self.binAmount*binWidth,y], width=1)
            t = self.app.fontSmaller.render(f"{i*tick_interval}", True, [255,255,255])
            self.app.screen.blit(t, [bin_x - t.get_size()[0] - 10, y - t.get_size()[1]/2])

            

        for x in range(self.binAmount):
            i = DATA["counts"][x] / particleMax * self.animTick
            iInv = (1-i)**0.5

            binVal = DATA["bin_centers"][x]

            if binVal < 0.1:
                binVal *= 1000
                unit = "nm"
            else:
                unit = "um"

            pygame.draw.rect(self.app.screen, [255,165 + 90*iInv,255*iInv], [bin_x + x*binWidth, 500 - i*200, binWidth-2, i*200])
            t = self.app.fontSmaller.render(f"{binVal:.1f} {unit}", True, [255,255,255])
            t = pygame.transform.rotate(t, 45)
            self.app.screen.blit(t, [bin_x + (x + 0.5)*binWidth - t.get_size()[0]/2, 510])

            r2 = pygame.Rect(bin_x + x*binWidth, 500 - 200, binWidth, 200)
            if r2.collidepoint(self.app.mouse_pos):
                underMouse = x

        if underMouse != None:
            t = self.app.fontSmall.render(f"{DATA['counts'][underMouse]} Particles", True, [255,255,255])
            self.app.screen.blit(t, [bin_x + underMouse*binWidth - t.get_size()[0]/2, 500 + 55])

            binVal1 = DATA["bin_edges"][underMouse]
            binVal2 = DATA["bin_edges"][underMouse+1]

            if binVal1 < 0.1:
                binVal1 *= 1000
                unit1 = "nm"
            else:
                unit1 = "um"
            
            if binVal2 < 0.1:
                binVal2 *= 1000
                unit2 = "nm"
            else:
                unit2 = "um"

            t = self.app.fontSmall.render(f"Range: {binVal1:.1f}{unit1} - {binVal2:.1f}{unit2}", True, [255,255,255])
            self.app.screen.blit(t, [bin_x + underMouse*binWidth - t.get_size()[0]/2, 500 + 80])

        for i, x in enumerate(self.additionaldata):
            t = self.app.fontSmall.render(f"{x}: {self.additionaldata[x]}", True, [255,255,255])
            
            r = t.get_rect()
            r.topleft = [bin_x + 10, 500 + 100 + i*20]

            if r.collidepoint(self.app.mouse_pos):
                t = self.app.fontSmall.render(f"{x}: {self.additionaldata[x]}", True, [255,0,0])
                #self.app.screen.blit(t, [bin_x + 10, 500 + 100 + i*20])
                if "mouse0" in self.app.keypress:
                    pyperclip.copy(f"{self.additionaldata[x]}")
                    self.app.notify(f"Copied {x} to clipboard")

            self.app.screen.blit(t, [bin_x + 10, 500 + 100 + i*20])
            
        
        if self.app.buttonGoBackDisplay.tick():
            self.app.SAMPLEDISPLAY = None

        if self.app.buttonToggleArea.tick():
            self.dispAreaInfo = not self.dispAreaInfo
            self.animTick = 0


    def viewData(self):
        if self.particleData:
            self.app.SAMPLEDISPLAY = self
            self.animTick = 0
        else:
            self.app.notify("No data present")
            self.app.SAMPLEDISPLAY = None

    
    def tick(self):
        
        r = self.t.get_rect()
        r.topleft = self.rect.topleft

        for x in self.images:
            x.tick(self)


        if r.collidepoint(self.app.mouse_pos):
            self.app.screen.blit(self.tr, self.rect.topleft)
            self.drawInfo()

            
            if self.text != "Unsorted" and "mouse2" in self.app.keypress:
                self.app.contextMenuShow.clear()
                for x in self.contextMenus:
                    x.toggleOn(self.rect.topleft)

                #try:
                #    self.app.SAMPLEDISPLAY = self
                #    self.animTick = 0
                #except:
                #    self.app.SAMPLEDISPLAY = None
        else:
            self.app.screen.blit(self.t, self.rect.topleft)
        

def pipelineTick(app: "App"):

    if app.SAMPLEDISPLAY:
        app.SAMPLEDISPLAY.displayInfo()

    elif app.pipelineInspect:
        inspect(app)
    else:

        pipelineTickMenu(app)

def renderInspectImage(app: "App"):
    im = app.InspectImages[0]

    surf = im.resultInspectSurface if app.inspectToggle else im.InspectSurface

    # Centered horizontal blit position
    offset_x = app.res[0] / 2 - surf.get_width() / 2
    offset_y = 0

    app.screen.blit(surf, (offset_x, offset_y))

    if not im.PARTICLES:
        return

    # Original and display dimensions
    w0, h0 = im.resultarray.shape[:2]
    w1, h1 = surf.get_size()

    sx = w1 / w0
    sy = h1 / h0

    for particle in im.PARTICLES:
        if not isinstance(particle.coords, np.ndarray):
            continue
        y = particle.coords[:, 1]
        x = particle.coords[:, 0]

        # Scale
        xs = x * sx
        ys = y * sy

        # Apply blit offset
        xs = (xs + offset_x).astype(int)
        ys = (ys + offset_y).astype(int)

        pts = list(zip(xs, ys))

        color = [255,0,0] if particle.real_diameter >= 5 else [0,255,0]

        if len(pts) > 1:
            pts.append(pts[0])  # Close the loop
            pygame.draw.lines(app.screen, color, False, pts, 2)
        else:
            pygame.draw.circle(app.screen, color, pts[0], 2)



def inspect(app: "App"):
    im = app.InspectImages[0]

    renderInspectImage(app)

    t = app.font.render(im.key, True, [255,255,255])
    app.screen.blit(t, [20,100])

    if app.buttonGoBackInspect.tick():
        app.pipelineInspect = False

    if app.buttonToggleInspect.tick():
        app.inspectToggle = not app.inspectToggle

    if app.buttonUse.tick():
        im.useInDispersionAnalysis = "use"
        app.InspectImages.remove(im)

    if app.buttonDontUse.tick():
        im.useInDispersionAnalysis = "dontuse"
        app.InspectImages.remove(im)
        app.inspectToggle = True

    if not app.InspectImages:
        app.pipelineInspect = False

def genInspect(app: "App"):
    app.InspectImages = []
    for x in app.SAMPLEDATA:
        if x == "Unsorted":
            continue

        for y in app.SAMPLEDATA[x].images:
            if not y.resultsurface:
                continue 

            if y.useInDispersionAnalysis == "undefined":
                app.InspectImages.append(y)

    
def setButtons(app: "App"):
    app.buttonInspect.locked = True
    app.buttonDisp.locked = True

    app.buttonRun.locked = False
    tot = 0
    totUsable = 0
    types = []
    for x in app.SAMPLEDATA:
        if x == "Unsorted":
            continue
        for y in app.SAMPLEDATA[x].images:


            if isinstance(y.resultarray, np.ndarray):
                types.append(y.useInDispersionAnalysis)
                
            tot += 1


    if not tot:
        app.buttonRun.locked = True
        app.buttonInspect.locked = True
        app.buttonDisp.locked = True

    if "undefined" in types:
        app.buttonInspect.locked = False
    
    if "undefined" not in types and types:
        app.buttonDisp.locked = False

    if app.PIPELINEACTIVE:
        app.buttonDisp.locked = True
        app.buttonRun.locked = True
            
    


def pipelineTickMenu(app: "App"):
    if app.buttonAddSample.tick():
        text = simpledialog.askstring("Input", "Enter name of the sample:")
        if text not in app.SAMPLEDATA and text:
            w = 50
            for x in app.SAMPLEDATA:
                w += app.SAMPLEDATA[x].w
            app.SAMPLEDATA[text] = Sample(app, text, w)

    if app.buttonAddData.tick():
        app.imageLoadDir = app.get_value("imageLoadDir", "")
        file_paths = filedialog.askopenfilenames(title="Select Data", 
                                         filetypes=[("Data Files", "*.txt;*.tif")],
                                         initialdir=app.imageLoadDir)
        

        if file_paths:
            folderPath = os.path.dirname(os.path.abspath(file_paths[0]))
            print("FOLDER OF IMAGE", folderPath)
            app.imageLoadDir = folderPath
            app.set_value("imageLoadDir", app.imageLoadDir)
        else:
            print("No file selected.")

     

        file_pairs = sortData(file_paths)
        for x in file_pairs:
            try:
                ImageDataObj(app, x, file_pairs[x])
            except:
                traceback.print_exc()


    if app.buttonSaveData.tick():
        FILE = filedialog.asksaveasfile(
                mode='wb',
                defaultextension=".pkl",
                confirmoverwrite=True,
                filetypes=[("Pipeline files", "*.ppl")],  # Only allow .pkl files
                title="Save pipeline",
                initialdir=f"{app.MAINPATH}/pipelines"
            )
        if not FILE:
            return
        save_current_project(app, FILE)


    if app.buttonLoadData.tick():
        FILE = filedialog.askopenfile(
                mode='rb',
                defaultextension=".ppl",
                filetypes=[("Pipeline files", "*.ppl")],  # Only allow .pkl files
                title="Load pipeline",
                initialdir=f"{app.MAINPATH}/pipelines"
            )
        if not FILE:
            return
        
        load_project(app, FILE)

    
    setButtons(app)

    if app.PIPELINEACTIVE:
        r1 = pygame.Rect(450, 150, 800, 20)
        
        r2 = r1.copy()
        r1.inflate_ip(4,4)
        pygame.draw.rect(app.screen, [255,255,255], r1, width=1)
        progress = (app.PIPELINEPROGRESS + app.PIPELINESUBPROGRESS) / app.PIPELINETARGETPROGRESS
        r2.width = 800 * progress
        pygame.draw.rect(app.screen, [255,255,255], r2)

        if app.pipelineEstimate + 1 < time.time():
            delta = time.time() - app.pipelineEstimate
            progressDelta = progress - app.progressSave
            app.progressSave = progress
            if delta < 2 and progressDelta and delta:
                app.pipelineTimeToComplete  = (1 - progress) / (progressDelta/delta)
            else:
                app.pipelineTimeToComplete = 0

            app.pipelineEstimate = time.time()

        e = f"Estimated time: {app.pipelineTimeToComplete:.0f}s"

        t = app.fontSmaller.render(e, True, [255,255,255])
        app.screen.blit(t, [450, 135])

    if app.buttonDisp.tick():
        if not app.PIPELINEACTIVE:
            start_new_thread(CALCDISPERSION, (app, ))

    if app.buttonInspect.tick():
        genInspect(app)
        if app.InspectImages:
            app.pipelineInspect = True
            app.inspectToggle = True

    if app.buttonRun.tick():
        if not app.PIPELINEACTIVE:
            start_new_thread(PIPELINEWRAP, (app, ))


    dataPresent = False
    for x in app.SAMPLEDATA:
        sample = app.SAMPLEDATA[x]
        sample.tick()
        if sample.additionaldata:
            dataPresent = True
        if x not in app.SAMPLEDATA:
            break

    app.exportCSV.locked = not dataPresent and not app.PIPELINEACTIVE
    if app.exportCSV.tick():

        allSamples = list(app.SAMPLEDATA.keys())
        allSamples.remove("Unsorted")

        # Collect all keys across all samples
        all_keys = set().union(*(app.SAMPLEDATA[s].additionaldata.keys() for s in allSamples))

        # Write header
        csv_lines = [app.SAMPLEDATA[allSamples[0]].additionaldata_csv_header(keys=all_keys)]

        # Write data rows
        for s in allSamples:
            csv_lines.append(app.SAMPLEDATA[s].additionaldata_to_csv_row(keys=all_keys))

        csv_output = "\n".join(csv_lines)
        export_csv_with_dialog(csv_output)
        app.notify("CSV exported!")

    app.tickContextMenus()

def create_log_histogram(diameters, bins=10):
    """Create logarithmic histogram of particle diameters"""
    min_val = np.log10(np.min(diameters))
    max_val = np.log10(np.max(diameters))
    
    # Create logarithmically spaced bins
    log_bins = np.logspace(min_val, max_val, bins + 1)
    hist, bin_edges = np.histogram(diameters, bins=log_bins)
    
    return {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'bin_centers': np.sqrt(bin_edges[:-1] * bin_edges[1:]).tolist()
    }

def create_log_area_histogram(particles, bins=10):
    diameters = [particle.real_diameter for particle in particles if particle.real_diameter > 0]
    #areas = [particle.real_area for particle in particles]  # Convert diameters to areas

    min_val = np.log10(np.min(diameters))
    max_val = np.log10(np.max(diameters))
    # Create logarithmically spaced bins
    log_bins = np.logspace(min_val, max_val, bins + 1)
    hist = np.zeros(len(log_bins)-1)
    for x in particles:
        if x.real_diameter <= 0:
            continue
        for y in range(0, len(log_bins)-1):
            if x.real_diameter > log_bins[y] and x.real_diameter <= log_bins[y+1]:
                hist[y] += x.real_area
                break
        
    return {
        'counts': hist.tolist(),
        'bin_edges': log_bins.tolist(),
        'bin_centers': np.sqrt(log_bins[:-1] * log_bins[1:]).tolist()
    }

def CALCDISPERSION(app: "App"):
    app.PIPELINEACTIVE = True

    totalImages = 0
    for x in app.SAMPLEDATA:
        if x == "Unsorted":
            continue
        for y in app.SAMPLEDATA[x].images:
            if y.useInDispersionAnalysis != "use":
                continue
            totalImages += 1

    app.PIPELINEPROGRESS = 0
    app.PIPELINETARGETPROGRESS = totalImages
    app.PIPELINESUBPROGRESS = 0


    for x in app.SAMPLEDATA:
        print("DATA FOR:", x)
        if x == "Unsorted":
            continue

        app.notify(f"Dispersion pipeline processing sample {x}")

        SAMPLE = app.SAMPLEDATA[x]
        ims = 0
        for y in SAMPLE.images:
            if y.useInDispersionAnalysis != "use":
                continue



            ims += 1
            app.PIPELINERESULT = y
            y.getParticles()
                       
            app.PIPELINERESULT = None
            app.PIPELINEPROGRESS += 1
            app.PIPELINESUBPROGRESS = 0
        if ims:
            SAMPLE.compileDispersionData()
        




    
    app.PIPELINEACTIVE = False
    app.notify("Dispersion pipeline finished")

def plot_size_distribution(app: "App", sample_name):
    """Plot the particle size distribution for a sample"""
    import matplotlib.pyplot as plt
    
    hist_data = app.SAMPLEDATA[sample_name].dispersion["size_distribution"]
    
    plt.figure(figsize=(12, 6))
    plt.bar(hist_data["bin_centers"], hist_data["counts"], 
            width=np.diff(hist_data["bin_edges"]),
            alpha=0.7, align='center')
    
    plt.xscale('log')
    
    # Format x-axis labels
    bin_centers = hist_data["bin_centers"]
    plt.xticks(bin_centers, 
               [f'{x:.1f}' for x in bin_centers],
               rotation=45,
               ha='right')
    
    plt.xlabel('Particle Diameter (μm)')
    plt.ylabel('Count')
    plt.title(f'Particle Size Distribution - {sample_name}')
    plt.grid(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def PIPELINEWRAP(app: "App", customSelection = []):
    try:
        PIPELINE(app, customSelection)
    except:
        print("Error occurred in PIPELINE:")
        traceback.print_exc()


def PIPELINE(app: "App", customSelection = []):
    app.PIPELINEACTIVE = True

    if not customSelection:

        override = True
        for y in [y for x in app.SAMPLEDATA for y in app.SAMPLEDATA[x].images]:
            if y.useInDispersionAnalysis == "use":
                override = messagebox.askyesno("Override Images", "Would you like to override the used images?")
                break

        
        totalImages = 0
        for x in app.SAMPLEDATA:
            if x == "Unsorted":
                continue
            for y in app.SAMPLEDATA[x].images:
                
                if not override and y.useInDispersionAnalysis == "use":
                    continue

                totalImages += 1
    else:
        override = True
        totalImages = len(customSelection)

    app.PIPELINEPROGRESS = 0
    app.PIPELINETARGETPROGRESS = totalImages
    app.PIPELINESUBPROGRESS = 0



    for x in app.SAMPLEDATA:
        if x == "Unsorted":
            continue

        app.notify(f"Pipeline processing sample {x}")

        for i, y in enumerate(app.SAMPLEDATA[x].images):
            print(y.key)
            if customSelection and y not in customSelection:
                continue
            app.notify(f"{x}: Image {i+1}/{len(app.SAMPLEDATA[x].images)}")
            if y.useInDispersionAnalysis == "use" and not override:
                continue

            if y.INDIVIDUAL_NODES:
                app.CURR_NODES = y.INDIVIDUAL_NODES
            else:
                app.CURR_NODES = app.NODES



            app.clearCache()
            app.setBackground(y.surface)
            app.pipelineImage = y.surface
            app.PIPELINERESULT = y
            
            godNodes = []

            
            for g in app.CURR_NODES:
                g.calcedThisExport = False
                if g.godNode and not g.disabled:
                    godNodes.append(g)

            #print("GODNODES:", godNodes)


            app.multipleGodNodeCalc(godNodes)
            print(f"{y} calc done.")
            app.PIPELINERESULT = None
            app.PIPELINEPROGRESS += 1
            app.PIPELINESUBPROGRESS = 0

            
    app.PIPELINEACTIVE = False
    app.pipelineImage = None
    app.notify("AI Pipeline finished")

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

import pickle
import numpy as np
from typing import Dict, List, Any, Optional

class SerializableParticle:
    """Serializable version of Particle class"""
    def __init__(self, particle_pixels: int, diameter: float, real_area: float, real_diameter: float):
        self.particle_pixels = particle_pixels
        self.diameter = diameter
        self.real_area = real_area
        self.real_diameter = real_diameter

class SerializableImageDataObj:
    """Serializable version of ImageDataObj class"""
    def __init__(self, image_data_obj):
        # File paths
        self.tifPath = image_data_obj.tifPath
        self.txtPath = image_data_obj.txtPath
        self.key = image_data_obj.key
        
        # Metadata
        self.FOVVal = image_data_obj.FOVVal
        self.MAGNIFICATION = image_data_obj.MAGNIFICATION
        self.SCANROTATION = image_data_obj.SCANROTATION
        self.STAGEPOSITION = image_data_obj.STAGEPOSITION
        self.parsed = image_data_obj.parsed
        
        # Analysis results (if available)
        self.useInDispersionAnalysis = image_data_obj.useInDispersionAnalysis
        
        # Save numpy arrays if they exist
        self.resultarray = image_data_obj.resultarray if hasattr(image_data_obj, 'resultarray') else None
        #self.labelArray = getattr(image_data_obj, 'labelArray', None)
        self.pixelCounts = getattr(image_data_obj, 'pixelCounts', None)
        self.diameters = getattr(image_data_obj, 'diameters', None)
        
        # FOV data (assuming it's serializable - if not, you'll need to handle this separately)
        self.FOV = image_data_obj.FOV if image_data_obj.parsed else None
        
        # Calculated values
        self.pixel_length = getattr(image_data_obj, 'pixel_length', None)
        self.image_area = getattr(image_data_obj, 'image_area', None)
        self.pixel_area = getattr(image_data_obj, 'pixel_area', None)
        
        # Convert particles to serializable format
        self.particles_data = []
        if hasattr(image_data_obj, 'PARTICLES'):
            for particle in image_data_obj.PARTICLES:
                self.particles_data.append(SerializableParticle(
                    particle.particle_pixels,
                    particle.diameter,
                    particle.real_area,
                    particle.real_diameter
                ))

class SerializableSample:
    """Serializable version of Sample class"""
    def __init__(self, sample):
        self.text = sample.text
        self.xpos = sample.xpos
        self.URF = sample.URF
        self.animTick = sample.animTick
        self.dispAreaInfo = sample.dispAreaInfo
        
        # Convert images to serializable format
        self.images_data = []
        for image in sample.images:
            self.images_data.append(SerializableImageDataObj(image))
        
        # Analysis data
        self.particleData = getattr(sample, 'particleData', [])
        self.particleDataArea = getattr(sample, 'particleDataArea', [])
        self.unDispersedArea = getattr(sample, 'unDispersedArea', 0)
        self.totalArea = getattr(sample, 'totalArea', 0)
        self.minSize = getattr(sample, 'minSize', 0)
        self.maxSize = getattr(sample, 'maxSize', 0)
        self.areas = getattr(sample, 'areas', [])
        self.averageArea = getattr(sample, 'averageArea', 0)
        self.medianArea = getattr(sample, 'medianArea', 0)
        self.UNDISPERSEDPARTICLES = getattr(sample, 'UNDISPERSEDPARTICLES', 0)
        self.fillerarea = getattr(sample, 'fillerarea', 0)
        self.undispfillerarea = getattr(sample, 'undispfillerarea', 0)
        self.additionaldata = getattr(sample, 'additionaldata', {})

def save_project_data(sample_data: Dict, file):
    """Save project data to pickle file"""
    serializable_data = {}
    
    for sample_name, sample in sample_data.items():
        serializable_data[sample_name] = SerializableSample(sample)
    
    pickle.dump(serializable_data, file)
    print("Saved!")
    

def load_project_data(FILE, app) -> Dict:
    """Load project data from pickle file and reconstruct objects"""
    serializable_data = pickle.load(FILE)
    
    reconstructed_samples = {}
    
    for sample_name, serializable_sample in serializable_data.items():
        # Create new Sample object
        sample = Sample(app, serializable_sample.text, serializable_sample.xpos)
        
        # Restore sample properties
        sample.URF = serializable_sample.URF
        sample.animTick = serializable_sample.animTick
        sample.dispAreaInfo = serializable_sample.dispAreaInfo
        sample.particleData = serializable_sample.particleData
        sample.particleDataArea = serializable_sample.particleDataArea
        sample.unDispersedArea = serializable_sample.unDispersedArea
        sample.totalArea = serializable_sample.totalArea
        sample.minSize = serializable_sample.minSize
        sample.maxSize = serializable_sample.maxSize
        sample.areas = serializable_sample.areas
        sample.averageArea = serializable_sample.averageArea
        sample.medianArea = serializable_sample.medianArea
        sample.UNDISPERSEDPARTICLES = serializable_sample.UNDISPERSEDPARTICLES
        sample.fillerarea = serializable_sample.fillerarea
        sample.undispfillerarea = serializable_sample.undispfillerarea
        sample.additionaldata = serializable_sample.additionaldata
        
        # Clear the images list (it was populated in __init__)
        sample.images.clear()
        
        # Reconstruct ImageDataObj objects
        for serializable_image in serializable_sample.images_data:
            # Create new ImageDataObj
            image_dict = {
                "tif": serializable_image.tifPath,
                "txt": serializable_image.txtPath
            }
            image = ImageDataObj(app, serializable_image.key, image_dict)
            if not image.LOADED:
                continue
            
            # Restore properties
            
            image.resultarray = serializable_image.resultarray

            if isinstance(image.resultarray, np.ndarray):
                im2d = image.resultarray[:, :, 0]
                labels = measure.label(im2d)# Label connected components
                image.labelArray = labels  
            else:
                image.labelArray = None

            image.pixelCounts = serializable_image.pixelCounts
            image.diameters = serializable_image.diameters
            image.pixel_length = serializable_image.pixel_length
            image.image_area = serializable_image.image_area
            image.pixel_area = serializable_image.pixel_area
            
            # Reconstruct particles
            if serializable_image.particles_data:
                image.PARTICLES = []
                for particle_data in serializable_image.particles_data:
                    particle = Particle.__new__(Particle)  # Create without calling __init__
                    particle.particle_pixels = particle_data.particle_pixels
                    particle.diameter = particle_data.diameter
                    particle.real_area = particle_data.real_area
                    particle.real_diameter = particle_data.real_diameter
                    image.PARTICLES.append(particle)
            
            # Regenerate surfaces (these will be recreated from the image files)
            if image.resultarray is not None:
                image.addResult(image.resultarray)

            image.useInDispersionAnalysis = serializable_image.useInDispersionAnalysis
            
            # Remove from "Unsorted" and add to correct sample
            if image in app.SAMPLEDATA["Unsorted"].images:
                app.SAMPLEDATA["Unsorted"].images.remove(image)
            sample.images.append(image)
        
        # Regenerate image map
        if sample.text != "Unsorted":
            sample.genImageMap()
        
        reconstructed_samples[sample_name] = sample
    
    #print(f"Project data loaded from {filename}")
    return reconstructed_samples

# Usage example functions
def save_current_project(app, filename: str):
    """Save the current project state"""
    save_project_data(app.SAMPLEDATA, filename)
    app.notify("Project saved!")

def load_project(app, FILE):
    """Load a project and replace current data"""
    # Clear existing data
    app.SAMPLEDATA.clear()
    app.SAMPLEDATA = {"Unsorted": Sample(app, "Unsorted", 50)}
    # Load and reconstruct
    loaded_data = load_project_data(FILE, app)
    
    # Update app's sample data
    app.SAMPLEDATA.update(loaded_data)
    
    # Ensure "Unsorted" sample exists
    if "Unsorted" not in app.SAMPLEDATA:
        app.SAMPLEDATA["Unsorted"] = Sample(app, "Unsorted", 50)
    
    app.notify("Pipeline loaded!")







if __name__ == "__main__":
    
    print(a)