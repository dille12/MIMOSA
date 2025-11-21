
from core.funcs.pointCheck import is_point_between
from core.funcs.bezier import draw_bezier_curve
from core.funcs.toolTip import toolTip
import pygame
import math
import traceback

from core.keyhint import KeyHint


def fetchNodeIOtype(typeC):

    if str(typeC) == "<class 'int'>" or str(typeC) == "<class 'float'>" or str(typeC) == "int":
        rType = "Number"
    elif "ndarray" in str(typeC):
        rType = "Array"
    elif str(typeC) == "<class 'bool'>":
        rType = "Boolean"

    elif "pygame.surface.Surface" in str(typeC):
        rType = "Image"
    
    elif str(typeC) == "<class 'inspect._empty'>":
        rType = "ANY"
    elif "dict" in str(typeC):
        rType = "Dict"
    elif "list" in str(typeC):
        rType = "List"
    elif "FOV" in str(typeC):
        rType = "FOV"
    else:
        rType = "ANY"

    return rType



class InputNode:
    def __init__(self, parent, inputName, typeC, isOutput = False, nodeLog = None):
        self.h = parent.height
        self.nodeLog = nodeLog
        self.parent = parent
        self.input = inputName
        self.isOutput = isOutput
        self.app = parent.app
        self.selfInput = False
        self.nodeLog.add("RAW TYPE:", typeC, len(str(typeC)))
        self.typeC = fetchNodeIOtype(typeC)
        self.nodeLog.add("FILTERED:", self.typeC, len(self.typeC))

        nt = "Output" if self.isOutput else "Input"
        pt = "input" if self.isOutput else "output"

        tp = f"Connect this to an {pt} to establish a connection."

        self.toolTip = toolTip(self.app, self, nt, tp)
        #self.nameRollTick = self.app.GT(120, oneshot = True)

        self.stashInput = None
        self.connectOnRelease = None


        self.connectKH = KeyHint(self.app, "LCLICK", "Start a connection")
        self.fConnectKH = KeyHint(self.app, "LCLICK", "Connect the nodes")
        self.deleteConnKH = KeyHint(self.app, "RCLICK", "Delete connection")
        self.cancelKH = KeyHint(self.app, "RCLICK", "Cancel")
        
        self.waveI = 0

        self.outputIndex = 0
        
        self.nodeLog.add("Creating input node for:", self.input)
        if self.input == "PARENT":
            self.selfInput = self.parent.cardParent
            self.nodeLog.add("Set as auto input: PARENT")

        if self.input == "Node":
            self.selfInput = self.parent
            self.nodeLog.add("Set as auto input: NODE")

        if self.input == "App":
            self.selfInput = self.app

        if not self.selfInput:
            parent.height += 30

        self.CONNECTION = None

        self.IN_CONNECTION = None

        if self.isOutput:
            self.x = self.parent.WIDTH - 10
            self.parent.outputs.append(self)

            self.outputIndex = self.parent.outputs.index(self)

            self.nodeLog.add("IS OUTPUT")
        else:
            self.x = 10
            self.parent.inputs.append(self)

        self.pos = parent.app.v2(self.x, self.h)

    def __repr__(self):
        return self.parent.name

        

    def getValue(self, usedStash):

        if self.selfInput:
            return self.selfInput, usedStash
        
        if not self.IN_CONNECTION:
            print("Error. No connection")
            return "ERROR", usedStash
        try:
            RESULT, usedStash = self.IN_CONNECTION.parent.calc(usedStash)

            if self.IN_CONNECTION.parent.hasMultipleOutputs:
                print(type(RESULT))
                RESULT = RESULT[self.IN_CONNECTION.outputIndex]

        except:
            print("Error.")
            
            traceback.print_exc()
            return "ERROR", usedStash
        
        if self.stashInput != self.IN_CONNECTION:
            usedStash = False
        
        self.stashInput = self.IN_CONNECTION
        return RESULT, usedStash
    
    def checkInCompatability(self, node):
        return self.typeC != node.typeC and self.typeC != "ANY" and node.typeC != "ANY"
    
    def connect(self, node):
        self.nodeLog.add("Connecting:", self, "and", node)
        if self.checkInCompatability(node):
            self.nodeLog.add("WRONG TYPE")
            return
        
        if self.parent == node.parent:
            self.nodeLog.add("SELF CONNECT")
            return
        

        
        if self.isOutput:
            node.IN_CONNECTION = self

        elif not self.isOutput:
            node.connect(self)

        self.nodeLog.add(self, self.IN_CONNECTION, self.isOutput)
        self.nodeLog.add(node, node.IN_CONNECTION, node.isOutput)

        self.parent.startFromBottomConnectionRefresh()
        #node.parent.refreshConnection()


        if self.isOutput:
            self.parent.nudgeStructure()

        self.app.EXPORT = True




    def disconnectFrom(self, node):
        
        self.IN_CONNECTION = None

    def fetchOutConnections(self):
        l = []
        for x in self.parent.parentList:
            for y in x.inputs:
                if self == y.IN_CONNECTION:
                    l.append(y)
        
        return l
                


    def disconnect(self):
        

        if self.isOutput:
            l = self.fetchOutConnections()
            self.nodeLog.add("OUT CONNECTIONS:", l)
            for x in l:
                x.disconnect()

            self.parent.startFromBottomConnectionRefresh()

            
        
        else:
            n = self.IN_CONNECTION
            self.IN_CONNECTION = None
            if n:
                n.parent.startFromBottomConnectionRefresh()
        
        self.app.EXPORT = True

    
    def checkIfUnderMouse(self):
        if self.selfInput:
            return

        POS = self.pos + self.parent.pos

        r = pygame.Rect(POS, [0,0])
        r.inflate_ip(12,12)

        return r.collidepoint(self.app.mouse_pos)
        

    def render(self):

        if self.selfInput:
            return

        POS = self.pos + self.parent.pos

        r = pygame.Rect(POS, [0,0])

        RADIUS = 12

        r.inflate_ip(RADIUS,RADIUS)

        if r.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES:
            w = 4
            self.app.mouseAvailable = False

            self.toolTip.render()

            

            if not self.app.activeInputNode:
                self.connectKH.active = True
            else:
                self.fConnectKH.active = True

            self.deleteConnKH.active = True

            if "mouse2" in self.app.keypress:
                self.disconnect()

            elif "mouse0" in self.app.keypress:
                if not self.app.activeInputNode:
                    self.app.activeInputNode = self 


                elif self.app.activeInputNode.isOutput != self.isOutput:

                    if self is self.app.activeInputNode:
                        self.disconnect()

                    else:
                        
                        self.connect(self.app.activeInputNode)

                        
                        self.app.activeInputNode = None
                else:
                    self.app.activeInputNode = None
        else:
            w = 2

        if self.CONNECTION:
            COLOR = [255,255,255]
        elif self.isOutput:
            COLOR = [0,255,0]
        else:
            COLOR = [0,255,255]

        trianglePoints = []
        radius = RADIUS - 7
        angle_offset = 0
        for i in range(3):
            # Draw a triangle pointing to the right
            angle = angle_offset + i * 2 * math.pi / 3
            x = POS[0] + radius * math.cos(angle) - 1
            y = POS[1] + radius * math.sin(angle) - 1
            trianglePoints.append((x, y))

        pygame.draw.polygon(self.app.screen, COLOR, trianglePoints)

        pygame.draw.circle(self.app.screen, COLOR, POS, 8, width = w)

        t = self.app.fontSmaller.render(str(self.input) if not self.isOutput else str(self.typeC), True, [255,255,255])
        if not self.isOutput:
            self.app.screen.blit(t, self.pos + self.parent.pos + [8, -4])
        else:
            self.app.screen.blit(t, self.pos + self.parent.pos + [-t.get_size()[0]-10, -4])

        if self.IN_CONNECTION and not self.isOutput:

            

            POS2 = self.IN_CONNECTION.pos + self.IN_CONNECTION.parent.pos

            if self.app.clicked and self.app.clicked != self.parent and self.app.clicked != self.IN_CONNECTION.parent:
                if is_point_between(self.app.mouse_pos, POS, POS2, threshold=25):
                    self.app.connectionUnderMouse = self
                elif self.app.connectionUnderMouse == self:
                    self.app.connectionUnderMouse = None

                if self.app.connectionUnderMouse == self:
                    w = 4
                else:
                    w = 2
            else:
                self.app.connectionUnderMouse = None
                w = 2


            draw_bezier_curve(self.app.screen, [255,255,255], POS, POS2, width = w, steps = 15, waveI = self.IN_CONNECTION.waveI, startNode=self.IN_CONNECTION.parent)
            #pygame.draw.line(self.app.screen, [255,255,255], POS, POS2, width=w)

        if self.app.MANIPULATENODES:
            self.suggestLinks(POS)
        
        if self.waveI > -4:
            self.waveI -= 0.45
            if self.waveI <= -4:
                for x in self.fetchOutConnections():
                    for y in x.parent.outputs:
                        y.waveI = 20
                self.waveI = -4

        

    def suggestLinks(self, POS):
        self.connectOnRelease = None
        if self.app.clicked == self.parent and not self.app.connectionUnderMouse:
            
            if self.isOutput:
                if self.fetchOutConnections():
                    return
                
            else:
                if self.IN_CONNECTION:
                    return

            suggestion = None
            suggestionDist = 1000

            for x in self.parent.parentList:
                for y in x.inputs + x.outputs:
                    
                    if y.isOutput == self.isOutput or self.parent == y.parent or y.selfInput or self.selfInput:
                        continue

                    if self.checkInCompatability(y):
                        continue

                    if y in self.parent.suggestions:
                        continue

                    if not y.isOutput and y.IN_CONNECTION:
                        continue


                    posIONode = y.pos + y.parent.pos
                    diff = pygame.math.Vector2(POS - posIONode)
                    if self.isOutput:
                        diff[0] *= -1


                    if abs(diff[1]) < 100 and 10 < diff[0] < 300:

                        if diff.length() < suggestionDist:
                            suggestion = y
                            suggestionDist = diff.length()

            if suggestion:
                self.parent.suggestions.append(suggestion)
                self.connectOnRelease = suggestion
                posIONode = suggestion.pos + suggestion.parent.pos
                draw_bezier_curve(self.app.screen, [0,255,0], POS, posIONode, width = 1, steps = 15)