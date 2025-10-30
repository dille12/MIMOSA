import pygame
import numpy as np
from pygame.math import Vector2 as v2

import time

class ImageDisplayCache:
    """
    Base class for optimized image display with caching.
    Inherit from this class and implement the required methods.
    """
    
    def __init__(self):
        # Cache variables
        self.cached_image = None
        self.cached_crop_rect = None
        self.cached_scale_factor = None
        self.cached_blit_pos = None
        
        # State tracking for cache invalidation
        self.last_zoom = None
        self.last_image_pos = None
        self.last_res = None
        self.last_separate_display = None
        self.imageVersion = 0
        self.lastImageVersion = -1
        self.image_cache_dirty = True
        
        # Floating-point position for smooth panning
        self.imagePos_float = None
        print("ImageDisplayCache initialized")

    def get_current_screen_and_res(self):
        if self.separateDisplay:
            return self.dualWindowSurf, self.v2(self.dualWindowSurf.get_size())
        else:
            return self.screen, self.res
        
    def should_manipulate(self):
        return self.mode == 1 or self.mode == 2 or (self.separateDisplay and self.dualWindow.focused)

    
    def _get_image_hash(self, image):
        """Generate hash for image change detection"""
        if image is None:
            return None
        try:
            return hash(image.get_buffer().raw)
        except:
            # Fallback to object ID if buffer access fails
            return id(image)

    def setBackground(self, image):
        self.imageApplied = image
        self.imageVersion += 1
        self.imageVersion = self.imageVersion % 1000  # Keep it manageable

    def displayImageCached(self):
        """Main display method with caching optimization"""
        # Get current display parameters
        SCREEN, RES = self.get_current_screen_and_res()
        MANIPULATE = self.should_manipulate()
        keypress = self.keypress
        keypress_held = self.keypress_held_down
        mouse_delta = self.mouseDelta
        mouse_pos = self.mouse_pos
        
        # Handle zoom input
        if MANIPULATE:
            
            self.zoom += self.wheelI
           


        # Handle panning input
        if "mouse1" in keypress_held and MANIPULATE:
            # Use cached scale factor, fallback to 1 if not available
            scale_factor = self.cached_scale_factor if self.cached_scale_factor else 1
            
            # Initialize float position if not set
            if self.imagePos_float is None:
                self.imagePos_float = v2(self.imagePos) if self.imagePos is not None else v2(0, 0)
            
            # Update floating-point position
            self.imagePos_float += mouse_delta / scale_factor
            
            # Convert to integer position for pygame.Rect
            self.imagePos = v2(int(self.imagePos_float.x), int(self.imagePos_float.y))
            self.image_cache_dirty = True
        
        # Get current image and calculate basic parameters
        current_image = self.imageApplied
        self.imageSize = v2(current_image.get_size())
        self.zoom = np.clip(self.zoom, -2, 15)
                
        # Check if we need to recalculate
        needs_recalc = (
            self.image_cache_dirty or
            self.last_zoom != self.zoom or
            self.last_image_pos != self.imagePos or
            self.last_res != RES or
            self.imageVersion != self.lastImageVersion or
            self.cached_image is None
        )
        
        if needs_recalc:

            self.lastImageVersion = self.imageVersion

            self._recalculate_image_display(RES, current_image)
            
            # Update cached values
            self.last_zoom = self.zoom
            self.last_image_pos = self.imagePos.copy() if hasattr(self.imagePos, 'copy') else self.imagePos
            self.last_res = RES.copy() if hasattr(RES, 'copy') else RES
            
            self.image_cache_dirty = False

            

        # Always blit the cached image
        SCREEN.blit(self.cached_image, self.cached_blit_pos)
        if needs_recalc:
            self.debugText("Recalculating image display")
        else:
            self.debugText("Using cached image display")

        # Update mouse position calculations (lightweight operations)
        self._update_mouse_calculations(mouse_pos)


    def getZoomFactor(self):
        """Calculate zoom factor based on current zoom level"""
        return 0.83 ** (self.zoom - 1)
    
    def _recalculate_image_display(self, RES, current_image):
        """Recalculate image display parameters and cache the result"""
        #zoom_factor = (1 * (16 - self.zoomDelta) / 15) ** 1.5
        zoom_factor = self.getZoomFactor()
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
                new_size[0] = new_size[1] * CANVASRATIO
        else:
            mod_zoom_dim[1] = self.imageSize[0] / CANVASRATIO
            if new_size[0] / CANVASRATIO > self.imageSize[1]:
                new_size[1] = self.imageSize[1]
            else:
                cropped_inside = 1
                new_size[1] = new_size[0] / CANVASRATIO
        
        xDim, yDim = new_size

        




        
        # Calculate crop rectangle
        self.crop_rect = pygame.Rect(self.imageSize / 2 - self.imagePos, [0, 0])
        self.crop_rect.inflate_ip(xDim, yDim)
        
        self.crop_rect.x = max(0, self.crop_rect.x)
        self.crop_rect.y = max(0, self.crop_rect.y)
        self.crop_rect.x = min(self.crop_rect.x, self.imageSize[0] - xDim)
        self.crop_rect.y = min(self.crop_rect.y, self.imageSize[1] - yDim)

        if self.last_zoom != self.zoom and hasattr(self, 'IMBLITPOS'):
            mousePosReal = self.viewportToPixel(self.mouse_pos)
            z = zoom_factor
            currSize = new_size
            oldZoomPos = self.cached_crop_rect.topleft  # ζ in your equation
            oldSize = self.cached_crop_rect.size    # ρ in your equation

            self.crop_rect.x = get_zoomed_start(mod_zoom_dim[0], z, oldZoomPos[0], oldSize[0], mousePosReal[0])
            self.crop_rect.y = get_zoomed_start(mod_zoom_dim[1], z, oldZoomPos[1], oldSize[1], mousePosReal[1])


        self.crop_rect.x = max(0, self.crop_rect.x)
        self.crop_rect.y = max(0, self.crop_rect.y)
        self.crop_rect.x = min(self.crop_rect.x, self.imageSize[0] - xDim)
        self.crop_rect.y = min(self.crop_rect.y, self.imageSize[1] - yDim)

        
        # Update image position based on final crop rect
        new_image_pos = self.imageSize / 2 - self.crop_rect.center
        
        # Update both integer and floating-point positions
        self.imagePos = new_image_pos
        if self.imagePos_float is not None:
            # Only update float position if it differs significantly from the corrected position
            # This prevents the correction from interfering with smooth panning
            diff = new_image_pos - self.imagePos_float
            if abs(diff.x) > 1 or abs(diff.y) > 1:
                self.imagePos_float = v2(new_image_pos)
        
        # Create and scale the image
        
        self.SCALEFACTOR = min(RES[0] / self.crop_rect.size[0], RES[1] / self.crop_rect.size[1])

        try:
            TEMPIM = current_image.subsurface(self.crop_rect).copy()
        except:
            TEMPIM = current_image.copy()
            self.crop_rect = current_image.get_rect()

        TEMPIM = pygame.transform.scale_by(TEMPIM, self.SCALEFACTOR)
        
        # Cache the results
        self.cached_image = TEMPIM
        self.cached_crop_rect = self.crop_rect.copy()
        self.cached_scale_factor = self.SCALEFACTOR
        self.cached_blit_pos = RES / 2 - v2(TEMPIM.get_size()) / 2
        self.IMBLITPOS = self.cached_blit_pos
    
    def _update_mouse_calculations(self, mouse_pos):
        """Update mouse position calculations (lightweight operations)"""
        MPIM = mouse_pos - self.IMBLITPOS
        MPRATIO = MPIM / self.SCALEFACTOR
        self.topLeft = v2(self.crop_rect.x, self.crop_rect.y)
        self.mousePosIm = self.viewportToPixel(mouse_pos)
        self.debugText(str(self.crop_rect))
        self.debugText(str(self.imagePos))
    
    def invalidate_image_cache(self):
        """Call this method to force recalculation on next display"""
        self.image_cache_dirty = True
    
    def _values_changed_significantly(self, old_val, new_val, tolerance=0.01):
        """Check if values changed enough to warrant recalculation"""
        if hasattr(old_val, '__len__'):  # Vector-like
            return any(abs(a - b) > tolerance for a, b in zip(old_val, new_val))
        else:  # Scalar
            return abs(old_val - new_val) > tolerance
        


def get_zoomed_start(total_width, desired_zoom_factor, current_start, current_width, zoom_origin):
    """
    Calculate the new start position for zooming while keeping zoom_origin in the same relative position
    
    Args:
        total_width: Total width of the image/viewport
        desired_zoom_factor: Zoom factor to apply
        current_start: Current crop start position
        current_width: Current crop width
        zoom_origin: The point that should remain stationary during zoom
    
    Returns:
        new_start: New start position for the zoomed crop
    """
    # Calculate the ratio of the zoom origin to the current start and end points
    zoom_ratio = (zoom_origin - current_start) / current_width
    # Calculate the new width of the zoomed crop
    new_width = total_width * desired_zoom_factor
    # Calculate the new start position based on keeping zoom_origin in the same relative position
    new_center = zoom_origin
    start = new_center - new_width * zoom_ratio
    return start