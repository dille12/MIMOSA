import pygame
import time
import threading
import sys
import os
from loadLock import *
os.environ['SDL_VIDEO_CENTERED'] = '1'
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

    if progress > max_progress:
        return True
    
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

            fade_progress = fade_progress**2
            
            # Interpolate from red (255, 0, 0) to white (255, 255, 255)
            red_component = 255
            green_component = int(165 + 90 * fade_progress)
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
    return False

def initialize_app(app_container):
    from main import App
    app_container.append(App())

def main():
    pygame.init()
    infoObject = pygame.display.Info()
    fullRes = pygame.math.Vector2(infoObject.current_w, infoObject.current_h)

    app_container = []
    app_thread = threading.Thread(target=initialize_app, args=(app_container,))
    app_thread.start()

    screen_width, screen_height = 400, 400
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.NOFRAME)
    image = pygame.image.load("core/ICON.png").convert_alpha()
    image = pygame.transform.scale(image, [400, 400])

    pygame.display.set_caption("MIMOSA")
    pygame.display.set_icon(pygame.image.load("core/ICON.png").convert_alpha())

    splash_color = (0, 0, 0)
    font = pygame.font.Font("core/terminal.ttf", 60)
    font2 = pygame.font.Font("core/terminal.ttf", 20)
    text_surface = font.render("MIMOSA", True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2))

    progressTick = 0

    clock = pygame.time.Clock()
    while app_thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(splash_color)
        screen.blit(image, [0, 0])

        

        #screen.blit(text_surface, text_rect)

        if raster_scan_reveal(screen, text_surface, progressTick, (screen_width // 2 - text_rect.width // 2, screen_height // 2 - text_rect.height // 2), kernel_size=20):
            screen.blit(text_surface, text_rect)

        x = font2.render(LOADLOCKSTATE["load_point"], True, (155, 155, 155))
        x_rect = x.get_rect(center=(screen_width // 2, screen_height // 2 + 100))

        if pygame.key.get_focused():
            progressTick += 1
        #print(progressTick)

        screen.blit(x, x_rect)
        pygame.display.update()
        clock.tick(60)

    App = app_container[0]

    App.fullres = fullRes

    App.initLoop()

if __name__ == "__main__":

    main()
    print("Software executed successfully!")
