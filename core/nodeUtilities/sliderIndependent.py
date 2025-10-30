import pygame
from pygame.math import Vector2 as v2

class sliderIndependent:
    def __init__(self, pos, app, minVal, maxVal, initVal, name):
        self.app = app
        self.minVal = minVal
        self.maxVal = maxVal
        self.value = initVal
        self.selected = False
        self.name = name

        self.pos = app.v2(pos)

        self.rect = pygame.Rect(self.pos[0], self.pos[1], 250, 30)
        print("SLIDER CREATED")


    def render(self):
        rVal = False
        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)

        if outerRim.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES:
            self.app.mouseAvailable = False
            w = 2
            if "mouse0" in self.app.keypress:
                self.selected = True
            
        else:
            w = 1

        if "mouse0" not in self.app.keypress_held_down:


            self.selected = False
            

        if self.selected:
            
            if "shift" in self.app.keypress_held_down:
                VAL = self.value
                self.value += self.app.mouseDelta[0] * 0.1
                self.value = min(self.maxVal, self.value)
                self.value = max(self.minVal, self.value)
                if VAL != self.value:
                    rVal = True

            else:
                VAL = (self.app.mouse_pos[0] - self.rect.topleft[0]) / self.rect.width
                VAL = min(VAL, 1)
                VAL = max(VAL, 0)
                SETVAL = self.minVal + VAL * (self.maxVal - self.minVal) 
                if SETVAL != self.value:

                    rVal = True

                
                self.value = SETVAL

            t = self.app.fontSmaller.render(f"PRESS SHIFT FOR FINE ADJUST", True, self.MAINCOLOR)
            self.app.screen.blit(t, self.app.mouse_pos + [0, -20])

        pygame.draw.rect(self.app.screen, [255,255,255], outerRim, width=w)


        valRect = self.rect.copy()
        valRect.width = self.rect.width * (self.value - self.minVal) / (self.maxVal - self.minVal)

        pygame.draw.rect(self.app.screen, [255,255,255], valRect)

        t = self.app.fontSmaller.render(f"{self.name}: {self.value:.1f}", True, [255,255,255])
        self.app.screen.blit(t, [self.rect.x, self.rect.y-15])

        return rVal
