import pygame
from pygame.math import Vector2 as v2

class DropDown:
    def __init__(self, parent, app, selections):
        self.app = app
        self.parent = parent
        self.name = selections[0]
        self.dropDownSelections = selections[1:] 
        self.value = 0
        self.y = parent.height
        parent.height += 50
        print("DropDown created")
        self.selected = False

        self.resetTick = self.app.GT(60, oneshot = True)
        self.resetTick2 = self.app.GT(60, oneshot = True, startMaxed = True)
        self.xCutoff = 0
        self.textState = 0

        self.scrollI = 0

    def checkIfUnderMouse(self):
        POS = self.parent.pos + [4, self.y]
        r = pygame.Rect([POS, [self.parent.WIDTH-10, 20]])

        return r.collidepoint(self.app.mouse_pos)


    def render(self, utilitySelectable):

        POS = self.parent.pos + [4, self.y]
        r = pygame.Rect([POS, [self.parent.WIDTH-10, 20]])


        if not self.selected:

            try:
                t = self.app.fontSmall.render(self.dropDownSelections[self.value], True, [255,255,255])


                self.drawText(t, r, [255,255,255], POS + [2, 0])
            except Exception as e:
                print(e)

            
            #self.app.screen.blit(t, )

            if r.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES and utilitySelectable:
                self.app.mouseAvailable = False
                w = 2

                if "mouse0" in self.app.keypress:
                    self.selected = True

            else:
                w = 1

            pygame.draw.rect(self.app.screen, [255,255,255], r, width=w)

        else:

            

            
            if self.app.MANIPULATENODES:
                if "wheelUp" in self.app.keypress:
                    self.scrollI -= 1
                elif "wheelDown" in self.app.keypress:
                    self.scrollI += 1

                if "esc" in self.app.keypress:
                    self.selected = False

            self.scrollI = min(self.scrollI, len(self.dropDownSelections) - 5)
            self.scrollI = max(self.scrollI, 0)

            visibleSelections = self.dropDownSelections[self.scrollI : min(self.scrollI + 5, len(self.dropDownSelections))]

            r.height = len(visibleSelections) * 20

            pygame.draw.rect(self.app.screen, [0,0,0], r)

            if len(self.dropDownSelections) > 5:
                r2 = r.copy()
                r2.width = 10
                r2.left = r.left - 15

                pygame.draw.rect(self.app.screen, [255,255,255], r2, width=1)

                scrollBarHeight = 5 / len(self.dropDownSelections)
                scrollBarPos = self.scrollI / len(self.dropDownSelections)

                r3 = pygame.Rect(r2.topleft, [8, scrollBarHeight * r2.height])

                r3.top = r2.top + scrollBarPos *  r2.height

                pygame.draw.rect(self.app.screen, [255,255,255], r3)
            


            for i, x in enumerate(visibleSelections):
                
                if i + self.scrollI == self.value:
                    c = [255,255,255]
                else:
                    c = [100,100,100]

                TPOS = POS + [2, 0 + i*20]

                r2 = pygame.Rect(TPOS - [2, 0], [self.parent.WIDTH-10, 20])

                if r2.collidepoint(self.app.mouse_pos):
                    self.app.mouseAvailable = False
                    c = self.app.MAINCOLOR


                    
                    
                    if "mouse0" in self.app.keypress:
                        self.selected = False
                        self.value = i + self.scrollI
                        self.app.EXPORT = True

                pygame.draw.rect(self.app.screen, [0,0,0], r2)
                pygame.draw.rect(self.app.screen, c, r2, width=1)

                try:

                    t = self.app.fontSmall.render(x, True, c)
                    self.app.screen.blit(t, TPOS)
                except:
                    pass

            pygame.draw.rect(self.app.screen, [255,255,255], r, width=1)
            

        t = self.app.fontSmaller.render(f"{self.name}:", True, [255,255,255])
        self.app.screen.blit(t, [r.x, r.y-15])

    def drawText(self, t, r, COLOR, pos):

        if r.width >= t.get_size()[0] or self.app.activeNode == self:
            self.app.screen.blit(t, pos)
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