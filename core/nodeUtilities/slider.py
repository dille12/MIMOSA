import pygame
from pygame.math import Vector2 as v2

class Slider:
    def __init__(self, parent, app, minVal, maxVal, initVal, name):
        self.app = app
        self.minVal = minVal
        self.maxVal = maxVal
        self.value = initVal
        self.parent = parent
        self.selected = False
        self.name = name

        self.pos = [4, parent.height]

        parent.height += 50

        self.rect = pygame.Rect(self.pos[0], self.pos[1], self.parent.WIDTH - 8, 20)
        print("SLIDER CREATED")

    def checkIfUnderMouse(self):
        self.rect.topleft = self.parent.pos + self.pos
        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)
        return outerRim.collidepoint(self.app.mouse_pos)
    

    def getStashVal(self):
        if self in self.parent.utility:
            i = self.parent.utility.index(self)
            if len(self.parent.stashUtilityVal) > i:
                return self.parent.stashUtilityVal[i]
            else:
                return None
        else:
            return None


    def render(self, utilitySelectable):
        rVal = False

        self.rect.topleft = self.parent.pos + self.pos

        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)

        if outerRim.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES and utilitySelectable:
            self.app.mouseAvailable = False
            w = 2
            if "mouse0" in self.app.keypress:
                self.selected = True
            
        else:
            w = 1

        if "mouse0" not in self.app.keypress_held_down:
            if self.selected:
                self.app.EXPORT = True
            self.selected = False
            

        if self.selected:
            
            if "shift" in self.app.keypress_held_down:
                VAL = self.value
                self.value += self.app.mouseDelta[0] * 0.1
                self.value = min(self.maxVal, self.value)
                self.value = max(self.minVal, self.value)
                if VAL != self.value:
                    self.app.EXPORT = True
                    rVal = True
            else:
                VAL = (self.app.mouse_pos[0] - self.rect.topleft[0]) / self.rect.width
                VAL = min(VAL, 1)
                VAL = max(VAL, 0)
                SETVAL = self.minVal + VAL * (self.maxVal - self.minVal) 
                if SETVAL != self.value:

                    rVal = True

                    self.app.EXPORT = True
                
                self.value = SETVAL

            if self.value != self.getStashVal():
                self.app.EXPORT = True

            t = self.app.fontSmaller.render(f"PRESS SHIFT FOR FINE ADJUST", True, [255,0,0])
            self.app.screen.blit(t, self.app.mouse_pos + [0, -20])

        pygame.draw.rect(self.app.screen, [255,255,255], outerRim, width=w)


        valRect = self.rect.copy()
        valRect.width = self.rect.width * (self.value - self.minVal) / (self.maxVal - self.minVal)

        pygame.draw.rect(self.app.screen, [255,255,255], valRect)

        t = self.app.fontSmaller.render(f"{self.name}: {self.value:.1f}", True, [255,255,255])
        self.app.screen.blit(t, [self.rect.x, self.rect.y-15])

        return rVal

