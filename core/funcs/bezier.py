import pygame
import math 

def bezier_curve(p0, p1, p2, p3, t):
    """
    Calculate a point on a cubic Bezier curve.
    
    :param p0: Start point
    :param p1: First control point
    :param p2: Second control point
    :param p3: End point
    :param t: Parameter (0 <= t <= 1)
    :return: Point on the Bezier curve
    """
    x = (1-t)**3 * p0[0] + 3 * (1-t)**2 * t * p1[0] + 3 * (1-t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1-t)**3 * p0[1] + 3 * (1-t)**2 * t * p1[1] + 3 * (1-t) * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)

def draw_bezier_curve(screen, color, start, end, steps=100, width = 1, waveI = 0, startNode = None):
    """
    Draw a cubic Bezier curve between two points with two control points in the middle.

    :param screen: Pygame screen to draw on
    :param color: Color of the curve
    :param start: Start point (x, y)
    :param end: End point (x, y)
    :param steps: Number of segments to approximate the curve
    """
    # Calculate control points
    mid_x = (start[0] + end[0]) / 2
    control1 = (mid_x, start[1])
    control2 = (mid_x, end[1])

    # Generate points on the curve
    points = [bezier_curve(start, control1, control2, end, t) for t in [i / steps for i in range(steps + 1)]]

    if waveI == -4:
        waveI = -10

    waveI = waveI - 1  # Adjust waveI to be the index in the points list

    # Draw the curve
    for i in range(len(points) - 1):

        colorMod = abs(i - waveI) * 0.25
        colorMod = min(colorMod, 1.0)
        colorModInv = 1 - colorMod
        if startNode and startNode.CONNECTED:
            colorInv = (20,155,20)
        else:
            colorInv = (155,20,20)

        # Fade the color based on distance from waveI to red
        c = (
            int(color[0] * colorMod + colorInv[0] * colorModInv),
            int(color[1] * colorMod + colorInv[1] * colorModInv),
            int(color[2] * colorMod + colorInv[2] * colorModInv)
        )
        
        pygame.draw.line(screen, c, points[i], points[i+1], width)

# Example usage
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((30, 30, 30))

        # Draw a Bezier curve
        start_point = (200, 300)
        end_point = (600, 300)
        draw_bezier_curve(screen, (255, 0, 0), start_point, end_point)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
