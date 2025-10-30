import pygame
import numpy as np
from pygame import gfxdraw

class rectRenderer:
    def __init__(self, image_path, rect_size=10):
        """
        Initialize the renderer with an image path and rectangle size.
        
        Args:
            image_path: Path to the image file (PNG with transparency recommended)
            rect_size: Size of each rectangle representing a pixel
        """
        pygame.init()
        
        # Load the image
        self.original_image = pygame.image.load(image_path).convert_alpha()
        self.rect_size = rect_size
        
        # Convert to pixel array and then to numpy
        self.pixel_array = pygame.surfarray.array3d(self.original_image)
        self.alpha_array = pygame.surfarray.array_alpha(self.original_image)
        
        # Get dimensions
        self.width, self.height = self.pixel_array.shape[:2]
        
        # Create binary mask for white/visible pixels
        # Check for white pixels (RGB close to 255,255,255) AND not transparent
        white_mask = np.all(self.pixel_array >= 240, axis=2)  # Close to white
        alpha_mask = self.alpha_array > 128  # Not transparent
        self.binary_mask = white_mask & alpha_mask
        
        # Pre-calculate rectangle positions for performance
        self.white_pixel_positions = np.argwhere(self.binary_mask)
        
        print(f"Image loaded: {self.width}x{self.height}")
        print(f"White pixels found: {len(self.white_pixel_positions)}")
    
    def render_individual_rects(self, screen, offset_x=0, offset_y=0, color=(255, 255, 255)):
        """
        Render each white pixel as an individual rectangle.
        Good for small images or when you need per-pixel control.
        """
        for x, y in self.white_pixel_positions:
            rect = pygame.Rect(
                offset_x + x * self.rect_size,
                offset_y + y * self.rect_size,
                self.rect_size,
                self.rect_size
            )
            pygame.draw.rect(screen, color, rect)
    
    def render_batched_rects(self, screen, offset_x=0, offset_y=0, color=(255, 255, 255)):
        """
        Render rectangles in batches for better performance.
        """
        rects = []
        for x, y in self.white_pixel_positions:
            rect = pygame.Rect(
                offset_x + x * self.rect_size,
                offset_y + y * self.rect_size,
                self.rect_size,
                self.rect_size
            )
            rects.append(rect)
        
        # Draw all rectangles at once
        for rect in rects:
            pygame.draw.rect(screen, color, rect)
    
    def render_optimized_surface(self, screen, offset_x=0, offset_y=0, color=(255, 255, 255)):
        """
        Most efficient method: pre-render to a surface and scale it.
        Best for static images that don't change often.
        """
        if not hasattr(self, '_cached_surface') or self._cached_color != color:
            # Create a surface with the scaled dimensions
            scaled_width = self.width * self.rect_size
            scaled_height = self.height * self.rect_size
            self._cached_surface = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
            self._cached_surface.fill((0, 0, 0, 0))  # Transparent
            
            # Draw rectangles to the cached surface
            for x, y in self.white_pixel_positions:
                rect = pygame.Rect(
                    x * self.rect_size,
                    y * self.rect_size,
                    self.rect_size,
                    self.rect_size
                )
                pygame.draw.rect(self._cached_surface, color, rect)
            
            self._cached_color = color
        
        # Blit the pre-rendered surface
        screen.blit(self._cached_surface, (offset_x, offset_y))
    
    def render_numpy_optimized(self, screen, offset_x=0, offset_y=0, color=(255, 255, 255)):
        """
        Ultra-fast method using numpy and surfarray for large images.
        """
        if not hasattr(self, '_numpy_surface') or self._numpy_color != color:
            # Create expanded array
            expanded_array = np.repeat(np.repeat(self.binary_mask, self.rect_size, axis=0), 
                                     self.rect_size, axis=1)
            
            # Create RGB array
            scaled_height, scaled_width = expanded_array.shape
            rgb_array = np.zeros((scaled_width, scaled_height, 3), dtype=np.uint8)
            
            # Set color where mask is True
            rgb_array[expanded_array.T] = color
            
            # Create surface from array
            self._numpy_surface = pygame.surfarray.make_surface(rgb_array)
            self._numpy_color = color
        
        screen.blit(self._numpy_surface, (offset_x, offset_y))
    
    def change_rect_size(self, new_size):
        if new_size == self.rect_size:
            return
        """Change the rectangle size and clear caches."""
        self.rect_size = new_size
        # Clear cached surfaces
        if hasattr(self, '_cached_surface'):
            del self._cached_surface
        if hasattr(self, '_numpy_surface'):
            del self._numpy_surface
    
    def get_image_info(self):
        """Get information about the loaded image."""
        return {
            'original_size': (self.width, self.height),
            'scaled_size': (self.width * self.rect_size, self.height * self.rect_size),
            'white_pixels': len(self.white_pixel_positions),
            'rect_size': self.rect_size
        }
