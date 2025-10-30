import pygame

# Initialize Pygame
pygame.init()

# Set screen dimensions
WIDTH, HEIGHT = 800, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Recursive Visualization")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Font setup
pygame.font.init()
FONT = pygame.font.SysFont(None, 24)
FONT2 = pygame.font.SysFont(None, 30, bold = True)
class Step:
    def __init__(self, text, depth):
        self.text = text
        self.depth = depth
    
    def draw(self, y_offset):
        text_surface = FONT.render(self.text, True, WHITE)
        screen.blit(text_surface, (20 + self.depth * 200, 30 + y_offset))


class Box:
    def __init__(self, funcName):
        self.text = funcName


    def draw(self, depth):
        text_surface = FONT2.render(self.text, True, WHITE)
        screen.blit(text_surface, (20 + depth * 200, 20))

# Define the steps with indentation levels
steps = [
    Step("Input Fetching", 0),
    Step("Input Fetching (1/2)", 1),
    Step("Input Fetching", 2),
    Step("Input Fetching", 3),
    Step("Execution", 3),
    Step("Returnal", 3),
    Step("Execution", 2),
    Step("Returnal", 2),
    Step("Input Fetching (2/2)", 1),
    Step("Input Fetching", 2),
    Step("Input Fetching", 3),
    Step("Execution", 3),
    Step("Returnal", 3),
    Step("Execution", 2),
    Step("Returnal", 2),
    Step("Execution", 1),
    Step("Returnal", 1),
    Step("Execution", 0),
    Step("Returnal", 0),
]

funcs = [
    Box("export"),
    Box("math"),
    Box("lowerThreshold"),
    Box("upperThreshold"),
    Box("inputImage")
]


# Main loop
running = True
while running:
    screen.fill(BLACK)
    
    # Draw steps
    y = 20

    for i in range(len(funcs)):
        funcs[i].draw(i)

    for step in steps:
        step.draw(y)
        
        y += 30
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
