import pygame
from pygame.math import Vector2 as v2

class Slider:
    def __init__(self, app, minVal, maxVal, initVal, pos, f, args):
        self.app = app
        self.minVal = minVal
        self.maxVal = maxVal
        self.value = initVal
        self.pos = pos
        self.selected = False

        self.rect = pygame.Rect(self.pos[0], self.pos[1], 200, 20)

        self.f = f
        self.args = args


    def update(self):
        rVal = False
        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)

        if outerRim.collidepoint(self.app.mouse_pos):
            self.app.mouseAvailable = False
            w = 2
            if "mouse0" in self.app.keypress:
                self.selected = True
            
        else:
            w = 1

        if "mouse0" not in self.app.keypress_held_down:
            self.selected = False

        if self.selected:

            VAL = (self.app.mouse_pos[0] - self.pos[0]) / self.rect.width
            VAL = min(VAL, 1)
            VAL = max(VAL, 0)
            SETVAL = self.minVal + VAL * (self.maxVal - self.minVal) 
            if SETVAL != self.value:

                rVal = True

                #self.app.applyLuminosityMask(SETVAL)
            
            self.value = SETVAL

        pygame.draw.rect(self.app.screen, [255,255,255], outerRim, width=w)




        valRect = self.rect.copy()
        valRect.width = self.rect.width * (self.value - self.minVal) / (self.maxVal - self.minVal)

        pygame.draw.rect(self.app.screen, [255,255,255], valRect)

        return rVal

