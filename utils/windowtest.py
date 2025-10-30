import pygame
import math

def raster_scan_reveal(screen, source_surface, progress, position, kernel_size=5):
    """
    Reveals an image progressively using a raster scanning pattern with kernels.
    
    Args:
        screen: pygame surface to draw on
        source_surface: pygame surface containing the image to reveal
        progress: integer progress value (starts from 1)
        position: tuple (x, y) position where to draw the image
        kernel_size: size of each kernel (default 5)
    """
    if progress <= 0:
        return
    
    # Get dimensions
    width = source_surface.get_width()
    height = source_surface.get_height()
    
    # Calculate number of kernels in each dimension
    kernels_x = math.ceil(width / kernel_size)
    kernels_y = math.ceil(height / kernel_size)
    total_kernels = kernels_x * kernels_y
    
    # Each kernel progresses from 1x1 to kernel_size x kernel_size
    max_progress = total_kernels + kernel_size - 1
    
    # Clamp progress to valid range
    progress = min(progress, max_progress)
    
    # Create a temporary surface for the revealed image
    revealed_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    revealed_surface.fill((0, 0, 0, 0))  # Transparent background
    
    # Process each kernel
    for kernel_idx in range(total_kernels):
        # Each kernel starts when progress reaches kernel_idx + 1
        kernel_start_progress = kernel_idx + 1
        
        if progress < kernel_start_progress:
            continue  # This kernel hasn't started yet
        
        # Calculate kernel position in grid
        kernel_x = kernel_idx % kernels_x
        kernel_y = kernel_idx // kernels_x
        
        # Calculate pixel position of this kernel
        base_x = kernel_x * kernel_size
        base_y = kernel_y * kernel_size
        
        # Determine the size of the reveal for this kernel
        # Each kernel progresses through sizes 1, 2, 3, ... up to kernel_size
        ticks_since_start = progress - kernel_start_progress
        reveal_size = min(ticks_since_start + 1, kernel_size)
        
        # Calculate the center offset for smaller reveals
        offset = (kernel_size - reveal_size) // 2
        
        # Calculate the actual area to reveal
        reveal_x = base_x + offset
        reveal_y = base_y + offset
        
        # Make sure we don't go outside the source image bounds
        actual_width = min(reveal_size, width - reveal_x)
        actual_height = min(reveal_size, height - reveal_y)
        
        if actual_width > 0 and actual_height > 0:
            # Create a rect for the area to copy
            source_rect = pygame.Rect(reveal_x, reveal_y, actual_width, actual_height)
            
            # Calculate fade color (red to white based on reveal progress)
            fade_progress = (reveal_size - 1) / (kernel_size - 1) if kernel_size > 1 else 1.0
            fade_progress = max(0.0, min(1.0, fade_progress))  # Clamp to 0-1
            
            # Interpolate from red (255, 0, 0) to white (255, 255, 255)
            red_component = 255
            green_component = int(255 * fade_progress)
            blue_component = int(255 * fade_progress)
            fade_color = (red_component, green_component, blue_component)
            
            # Create a temporary surface for this kernel with the fade color
            kernel_surface = pygame.Surface((actual_width, actual_height), pygame.SRCALPHA)
            
            # Copy the source pixels
            kernel_surface.blit(source_surface, (0, 0), source_rect)
            
            # Apply color tint by creating a colored overlay
            color_overlay = pygame.Surface((actual_width, actual_height), pygame.SRCALPHA)
            color_overlay.fill(fade_color)
            
            # Blend the color with the source using multiply blend mode
            # For text surfaces (white/alpha), this will tint the white pixels
            kernel_surface.blit(color_overlay, (0, 0), special_flags=pygame.BLEND_MULT)
            
            # Draw the tinted kernel to the revealed surface
            revealed_surface.blit(kernel_surface, (reveal_x, reveal_y))
    
    # Draw the revealed surface to the screen
    screen.blit(revealed_surface, position)

# Example usage function
def demo_raster_scan():
    """
    Demonstration of the raster scan reveal function
    """
    pygame.init()
    
    # Create a demo text surface
    font = pygame.font.Font(None, 48)
    demo_surface = font.render("HELLO WORLD!", True, (255, 255, 255))
    demo_width, demo_height = demo_surface.get_size()
    
    # Setup display
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Raster Scan Reveal Demo")
    clock = pygame.time.Clock()
    
    kernel_size = 8
    progress = 0
    max_progress = math.ceil(demo_width / kernel_size) * math.ceil(demo_height / kernel_size) + kernel_size - 1
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    progress = (progress + 1) % (max_progress + 10)
                elif event.key == pygame.K_r:
                    progress = 0
                elif event.key == pygame.K_UP:
                    kernel_size = min(kernel_size + 1, 20)
                    max_progress = math.ceil(demo_width / kernel_size) * math.ceil(demo_height / kernel_size) + kernel_size - 1
                elif event.key == pygame.K_DOWN:
                    kernel_size = max(kernel_size - 1, 3)
                    max_progress = math.ceil(demo_width / kernel_size) * math.ceil(demo_height / kernel_size) + kernel_size - 1
        
        # Clear screen
        screen.fill((30, 30, 30))
        
        # Draw the original text for reference
        screen.blit(demo_surface, (50, 50))
        
        # Draw the raster scanned version
        raster_scan_reveal(screen, demo_surface, progress, (50, 150), kernel_size)
        
        # Draw progress info
        font_small = pygame.font.Font(None, 28)
        text = font_small.render(f"Progress: {progress}/{max_progress}", True, (255, 255, 255))
        screen.blit(text, (50, 280))
        
        kernel_text = font_small.render(f"Kernel Size: {kernel_size}", True, (255, 255, 255))
        screen.blit(kernel_text, (50, 310))
        
        instruction_text = font_small.render("SPACE: Next, R: Reset, UP/DOWN: Kernel size", True, (255, 255, 255))
        screen.blit(instruction_text, (50, 340))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    demo_raster_scan()