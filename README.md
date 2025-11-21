# MIMOSA

**MIMOSA** (Microscopy IMage prOcessing SoftwAre) is a **node-based image analysis toolkit** for accurate dispersion measurements for large datasets of **SEM images**. It provides a modular framework for particle segmentation and dispersion measurement, optimized for large image batches and research on **rubber–lignin composites**.

---

## Features
- Node-based visual pipeline for image processing  
- High-performance backend using **NumPy**, **SciPy**, **OpenCV**, **scikit-image**, and **Numba**  
- Real-time GUI built with **Pygame**
- Builtin support for **TensorFlow** semantic segmentation models
- Tools for thresholding, segmentation, and morphological analysis  
- Designed for **SEM images** and **particle dispersion studies**

---

## Installation
```bash
git clone https://github.com/dille12/MIMOSA.git
cd MIMOSA
pip install -r requirements.txt
```
On Linux, TKinter must be installed via bash:
```
sudo apt-get update
sudo apt-get install python3-tk
```
Download the neural network package from the Releases tab by clicking [here](https://github.com/dille12/MIMOSA/releases/tag/1.0).

