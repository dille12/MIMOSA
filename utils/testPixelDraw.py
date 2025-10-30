import pygame
import numpy as np
from pygame import gfxdraw

class ScalableImageRenderer:
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

# Example usage
def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Scalable Image Rectangles")
    clock = pygame.time.Clock()
    
    # Load your image (replace with your image path)
    try:
        renderer = ScalableImageRenderer("utils/ICONpixelmap.png", rect_size=5)
        
        # Print image info
        info = renderer.get_image_info()
        print(f"Original size: {info['original_size']}")
        print(f"Scaled size: {info['scaled_size']}")
        print(f"White pixels: {info['white_pixels']}")
        
    except pygame.error as e:
        print(f"Could not load image: {e}")
        print("Creating a test pattern instead...")
        
        # Create a test image if the file doesn't exist
        test_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
        test_surface.fill((0, 0, 0, 0))
        # Draw a simple pattern
        for i in range(0, 20, 2):
            for j in range(0, 20, 2):
                if (i + j) % 4 == 0:
                    pygame.draw.rect(test_surface, (255, 255, 255, 255), (i, j, 1, 1))
        
        pygame.image.save(test_surface, "test_pattern.png")
        renderer = ScalableImageRenderer("test_pattern.png", rect_size=10)
    
    # Rendering method selection
    current_method = 0
    methods = [
        ("Individual Rects", renderer.render_individual_rects),
        ("Batched Rects", renderer.render_batched_rects),
        ("Optimized Surface", renderer.render_optimized_surface),
        ("Numpy Optimized", renderer.render_numpy_optimized)
    ]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Cycle through rendering methods
                    current_method = (current_method + 1) % len(methods)
                    print(f"Switched to: {methods[current_method][0]}")
                elif event.key == pygame.K_UP:
                    # Increase rectangle size
                    new_size = min(renderer.rect_size + 1, 50)
                    renderer.change_rect_size(new_size)
                    print(f"Rectangle size: {new_size}")
                elif event.key == pygame.K_DOWN:
                    # Decrease rectangle size
                    new_size = max(renderer.rect_size - 1, 1)
                    renderer.change_rect_size(new_size)
                    print(f"Rectangle size: {new_size}")
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Render using current method
        method_name, method_func = methods[current_method]
        start_time = pygame.time.get_ticks()
        method_func(screen, offset_x=50, offset_y=50, color=(255, 255, 255))
        render_time = pygame.time.get_ticks() - start_time
        
        # Display info
        font = pygame.font.Font(None, 36)
        info_text = font.render(f"Method: {method_name} | Time: {render_time}ms | Size: {renderer.rect_size}", 
                               True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        controls_text = font.render("SPACE: Change method | UP/DOWN: Resize rectangles", 
                                  True, (200, 200, 200))
        screen.blit(controls_text, (10, 750))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()