import pygame 
import numpy as np
def configureMask(app, node):
    app.renderRect = None
    app.notify("Configuring Mask...")
    print(node)
    app.editingMask = node
    app.applyMask(node)

    
    app.mode = 1