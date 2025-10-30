import pygame
from pygame.math import Vector2 as v2
from tkinter import filedialog
from pathlib import Path

class fileInput:
    def __init__(self, parent, app, name):
        self.app = app
        self.value = ""
        self.parent = parent
        self.selected = False
        self.name = name

        self.pos = [4, parent.height]

        parent.height += 50

        self.rect = pygame.Rect(self.pos[0], self.pos[1], self.parent.WIDTH - 8, 20)
        print("INPUTBOX CREATED")
        self.resetTick = self.app.GT(60, oneshot = True)
        self.resetTick2 = self.app.GT(60, oneshot = True, startMaxed = True)
        self.xCutoff = 0
        self.textState = 0


    def drawText(self, t, r, COLOR, pos):

        if r.width >= t.get_size()[0] or self.app.activeNode == self:
            self.app.screen.blit(t, self.pos)
            return

        if self.resetTick.tick():

            if self.textState == 0:
                self.xCutoff += 0.4
                self.maxX = self.xCutoff
                if self.xCutoff + r.width >= t.get_size()[0]:
                    self.textState = 1
                    self.resetTick.value = 0

            elif self.textState == 1:
                self.resetTick2.value = 0
                self.resetTick.value = 0
                self.textState = 2

            elif self.textState == 2:
                self.xCutoff = (1 - self.resetTick2.ratio()) * self.maxX
                if self.resetTick2.tick():
                    self.xCutoff = 0
                    self.textState = 3


            elif self.textState == 3:
                self.textState = 0
                self.resetTick.value = 0



        self.app.screen.blit(t, pos, area = [self.xCutoff, 0, r.width, t.get_size()[1]])


    def checkIfUnderMouse(self):
        self.rect.topleft = self.parent.pos + self.pos
        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)
        return outerRim.collidepoint(self.app.mouse_pos)


    def render(self, utilitySelectable):
        rVal = False

        self.rect.topleft = self.parent.pos + self.pos

        outerRim = self.rect.copy()
        outerRim.inflate_ip(4,4)

        if outerRim.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES and utilitySelectable:
            self.app.mouseAvailable = False
            w = 2
            if "mouse0" in self.app.keypress:
                self.value = filedialog.askopenfilename(initialdir=self.app.imageLoadDir,
                                                        title = "Load data")
            
        else:
            w = 1



        path = Path(self.value)
        filename = path.name


        t = self.app.fontSmall.render(f"{filename}", True, [255,255,255])
        rText = pygame.Rect([0,0], self.rect.size)

        self.drawText(t, rText, [255,255,255], [self.rect.x, self.rect.y])

        
        #self.app.screen.blit(t, [self.rect.x, self.rect.y], area = rText)



        pygame.draw.rect(self.app.screen, [255,255,255], outerRim, width=w)


        t = self.app.fontSmaller.render(f"{self.name}", True, [255,255,255])
        self.app.screen.blit(t, [self.rect.x, self.rect.y-15])

        return rVal