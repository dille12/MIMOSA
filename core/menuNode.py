import pygame
from core.funcs.toolTip import toolTip
import colorsys


def hcl_to_rgb(hue, chroma, lightness):
    # Convert HCL to HSL approximation
    saturation = chroma / max(lightness, 1e-10)  # Avoid division by zero
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    
    # Scale RGB values to 0-255
    return tuple(int(c * 255) for c in (r, g, b))

class MenuNode:
    def __init__(self, app, sibling = None, parent = None, pos = None, text = "Test", returnNode = None, returnFunc = None, tooltip = None, colortint = None):
        self.app = app
        self.sibling = sibling
        self.parent = parent
        self.children = []
        self.siblings = []
        self.text = text
        self.returnNode = returnNode
        self.returnFunc = returnFunc
        self.tWidth = self.app.font.render(self.text, True, [0,0,0]).get_rect().width
        if self.returnNode:
            self.toolTip = toolTip(self.app, self, self.text, self.returnNode.NODEHELP, titleColor=self.returnNode.colortint)
        elif tooltip:
            self.toolTip = toolTip(self.app, self, self.text, tooltip)
        else:
            self.toolTip = None
        self.selected = False
        self.selectedPerm = False

        self.darkBG = None

        if self.text in app.nodeColorGrade:
            colortint = app.nodeColorGrade[self.text]
            
        else:
            colortint = None

        if colortint != None:
            self.colortint = hcl_to_rgb(colortint, 0.50, 0.65)
            print("COLORTINT:", self.colortint)
        elif self.parent:
            self.colortint = self.parent.colortint
        else:
            self.colortint = [255,255,255]
        

        if sibling:
            self.pos = sibling.pos + [0,35]
            self.addToSiblings(self.sibling)
            print(self.siblings)
            for x in self.siblings:
                self.tWidth = max(self.tWidth, x.tWidth)

            for x in self.siblings:
                x.tWidth = self.tWidth

        elif parent:
            
            self.parent.children.append(self)
            self.pos = self.parent.pos + [150, 0]
            
            for x in self.parent.children:
                self.addToSiblings(x)

        else:
            self.pos = self.app.v2(pos)

        

        self.app.menunodes.append(self)

        

    def addToSiblings(self, node):

        if node == self:
            return

        if node not in self.siblings:
            self.siblings.append(node)

        for x in node.siblings:
            if x not in self.siblings:
                self.addToSiblings(x)

        if self not in node.siblings:
            node.addToSiblings(self)


    def getSelfRect(self):
        tW = self.tWidth 
        if self.parent:
            tW = 0
            for x in self.parent.siblings + [self.parent]:
                tW = max(tW, x.tWidth)

            self.pos = self.parent.pos + [tW, 35 * self.parent.children.index(self)]


        wN = self.tWidth
        for x in self.siblings:
            wN = max(wN, x.tWidth)

        return pygame.Rect(self.pos, [wN,35])

    
    def render(self):

        
        
        self.r = self.getSelfRect()

        if not self.darkBG:
            self.darkBG = pygame.Surface(self.r.size).convert_alpha()
            self.darkBG.fill((0,0,0))
            self.darkBG.set_alpha(220)

            

        if self.parent:
            if not self.parent.selected:
                self.selected = False
                return
            
        if self.darkBG:
            self.app.screen.blit(self.darkBG, self.pos)
            
        

        if self.r.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES:
            self.app.mouseAvailable = False
            c = [255,200,200]
            w = 2

            if self.toolTip:
                self.toolTip.render()

            self.app.nodeUnderMouse = True

            self.selected = True

            for x in self.siblings:
                if x != self:
                    x.selected = False

            if "mouse0" in self.app.keypress:

                if self.returnNode:

                    n = self.returnNode.copy()
                    n.realPos = self.app.mouse_pos - [20,20]

                    n.addTo(self.app.CURR_NODES)
                    self.app.closeMenus()

                elif self.returnFunc:
                    self.returnFunc()
                        
        if self.selected:
            c = self.app.MAINCOLOR
            w = 1

        
        else:
            c = self.colortint
            self.selected = False
            w = 1
        pygame.draw.rect(self.app.screen, c, self.r, width=w)


        t = self.app.font.render(self.text, True, c)


        self.app.screen.blit(t, self.pos)
    