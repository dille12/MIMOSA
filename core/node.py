import core.nodeFuncs
import inspect
import pygame
import random
import time
#from pygame.math import Vector2 as v2
import traceback
import numpy as np
from core.nodeUtilities.slider import Slider
from _thread import start_new_thread
from core.funcs.pointCheck import is_point_between
from core.funcs.bezier import draw_bezier_curve
from core.funcs.toolTip import toolTip
from core.nodeBluePrint import nodeBluePrint
from core.inputNode import InputNode
from core.nodeUtilities.fileInput import fileInput
from core.nodeUtilities.dropDown import DropDown
from core.nodeUtilities.button import Button
import colorsys
from numba.core.registry import CPUDispatcher
from core.contextMenu import ContextMenu
def is_jitted(func):
    return isinstance(func, CPUDispatcher)

from core.keyhint import KeyHint

DROPDOWN = 1
SLIDER = 2
FILEINPUT = 3
BUTTON = 4

utilityMap = {
    DROPDOWN: "DROPDOWN",
    SLIDER: "SLIDER",
    FILEINPUT: "FILEINPUT",
    BUTTON: "BUTTON"
}



def hcl_to_rgb(hue, chroma, lightness):
    # Convert HCL to HSL approximation
    saturation = chroma / max(lightness, 1e-10)  # Avoid division by zero
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    
    # Scale RGB values to 0-255
    return tuple(int(c * 255) for c in (r, g, b))


class Log:
    def __init__(self):
        self.log = []

    def add(self, *args):
        for text in args:
            if isinstance(text, list):
                self.log += text
            else:
                self.log.append(text)


class Node:
    def __init__(self, *args):

        app, cardParent, name, inputs, output, f, UTILITIES, nodeType, NODEHELP, nodeLog = args

        self.nodeLog = nodeLog
        self.ARGS = args

        self.NODEHELP = NODEHELP
        self.name = name

        self.parentList = None


        self.moveKH = KeyHint(app, "RCLICK (HOLD)", "Move node")
        self.contextKH = KeyHint(app, "RCLICK", "Access context menu")
        self.disconnectKH = KeyHint(app, "SHIFT + RCLICK", "Disconnect node from all")
        #self.initCalcKH = KeyHint(app, "SHIFT + LCLICK", "Recalculate from this node")
        self.viewOutputKH = KeyHint(app, "SHIFT + LCLICK", "View this nodes output")
        self.disableKH = KeyHint(app, "CTRL + LCLICK", "Disable node")
        self.deleteKH = KeyHint(app, "DEL", "Delete node")

        self.waveTick = app.GT(90)

        self.CONNECTED = False

        self.height = 50
        self.WIDTH = 150
        self.inputs = []
        self.outputs = []
        self.UTILITIES = UTILITIES
        self.utility = []
        self.colortint = hcl_to_rgb(app.nodeColorGrade[nodeType], 0.50, 0.65)

        for x in UTILITIES:

            needsUtility = x[0]
            selections = x[1:]

            if needsUtility == DROPDOWN:
                self.utility.append(DropDown(self, app, selections))
            elif needsUtility == SLIDER:
                self.utility.append(Slider(self, app, float(selections[0]), float(selections[1]),float(selections[2]), selections[3]))
            elif needsUtility == FILEINPUT:
                self.utility.append(fileInput(self, app, str(selections[0])))
            elif needsUtility == BUTTON:
                self.utility.append(Button(self, app, str(selections[0]), str(selections[1])))

            

        self.f = f
        self.nodeLog.add("FUNCTION:", f)
        if is_jitted(f):
            self.nodeLog.add("JITTED FUNCTION")
        else:
            self.nodeLog.add("NOT JITTED FUNCTION")
        self.app = app
        self.cardParent = cardParent

        self.nodeType = nodeType

        self.hasMultipleOutputs = False
        self.usedStash = False
        self.stashUtilityVal = []
        self.STASHEDRESULT = None
        self.stashIm = None
        self.calcedThisExport = False
        

        ContextMenu(self, "Duplicate", lambda: self.duplicate())
        ContextMenu(self, "Delete", lambda: self.delete())


        self.disp = ""

        self.pos = app.v2([0, 0])
        self.realPos = app.v2([0, 0])

        for x in inputs:

            type = inputs[x].split(":")[-1]
            
            InputNode(self, x, type, False, self.nodeLog)

        if output:

            self.nodeLog.add("Creating the output")

            typeC =  output

            if "typing.Tuple" in str(typeC):
                self.nodeLog.add("MULTIPLE OUTPUTS")
                self.hasMultipleOutputs = True

                typeC = str(typeC).split("[")[1].split("]")[0]

                for x in typeC.split(","):
                    x = x.replace(" ", "")
                    
                    InputNode(self, x, x, True, self.nodeLog)

            else:
                
                InputNode(self, output, typeC, True, self.nodeLog)

            self.godNode = False
        else:
            self.nodeLog.add("GOD NODE")
            self.godNode = True

        self.nodeLog.add("NODE:", name)
        self.nodeLog.add("INPUTS:", inputs)
        self.nodeLog.add("OUTPUTS", output)

        self.initiatorNode = all(obj.selfInput for obj in self.inputs) and self.outputs
        
        
        if self.initiatorNode:
            self.nodeLog.add("INITATOR")
            print("INITIATOR NODE", self.name)

        self.highLightNode = None

        self.errored = True

        self.resetTick = self.app.GT(60, oneshot = True)
        self.resetTick2 = self.app.GT(60, oneshot = True, startMaxed = True)
        self.xCutoff = 0
        self.textState = 0

        self.disabled = False

        self.toolTip = toolTip(self.app, self, self.name, self.NODEHELP, titleColor = self.colortint)

        

        r = pygame.Rect((self.pos, (self.WIDTH, self.height)))

        self.surface = pygame.Surface(r.size, pygame.SRCALPHA).convert_alpha()


    
    def duplicate(self):
        n = self.copy()
        n.realPos = self.realPos.copy() + [20, 20]
        n.addTo(self.parentList)

    def addTo(self, addTo):
        addTo.append(self)
        self.parentList = addTo


    def copy(self):
        
        return Node(*self.ARGS)
    

    def save(self):

        nodeBluePrint(self, self.app)


    def drawName(self, r, COLOR):
        t = self.app.fontNode.render(self.name, True, COLOR)

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



        self.app.screen.blit(t, self.pos, area = [self.xCutoff, 0, self.WIDTH, t.get_size()[1]])

    def handleShortCuts(self):

        if not self.app.activeNode == self:
            return

        if isinstance(self.STASHEDRESULT, np.ndarray) or self.godNode:
            if "mouse0" in self.app.keypress and "shift" in self.app.keypress_held_down:
                if not self.godNode:
                    self.app.setBackground(pygame.surfarray.make_surface(self.STASHEDRESULT).convert())
                else:
                    self.initCalc()

        if "mouse2" in self.app.keypress and "shift" in self.app.keypress_held_down: 
            for x in self.inputs + self.outputs:
                x.disconnect()

        if "mouse0" in self.app.keypress and "ctrl" in self.app.keypress_held_down:
            self.disabled = not self.disabled
            self.nodeLog.add("Disabled toggle")
            self.clearCache()
            self.app.EXPORT = True



    def startFromBottomConnectionRefresh(self):
        if self.initiatorNode:
            self.refreshConnection()
            return

        for x in self.inputs:
            if x.IN_CONNECTION:
                x.IN_CONNECTION.parent.startFromBottomConnectionRefresh()

            else:
                x.parent.refreshConnection()


    def refreshConnection(self):

        b = self._checkIfConnected()

        self.CONNECTED = b
        return b
            
    def _checkIfConnected(self):
        for x in self.outputs:
            
            outConnections = x.fetchOutConnections()

            if not outConnections:
                return False

            for y in outConnections:
                b = y.parent.refreshConnection()
                if not b:
                    return False

        return True

    def render(self):


        if self.initiatorNode:
            if self.waveTick.tick():
                for x in self.outputs:
                    x.waveI = 20

        self.pos = 0.6 * self.pos + 0.4 * self.realPos

        self.handleShortCuts()

        self.highLightNode = None

        COLOR = self.colortint if not self.disabled else [155,100,100]


        if "mouse1" in self.app.keypress_held_down and self.app.MANIPULATENODES:
            self.pos += self.app.mouseDelta
            self.realPos += self.app.mouseDelta

        r = pygame.Rect((self.pos, (self.WIDTH, self.height)))

        if self.app.CALCING and self.app.CALCNODE == self:
            r2 = r.copy()
            r2.inflate_ip(8,8)
            pygame.draw.rect(self.app.screen, [0,255,255], r2, 1)

        if self.disabled:
            r2 = r.copy()
            r2.inflate_ip(8,8)
            pygame.draw.rect(self.app.screen, COLOR, r2, 1)

        w = 1

        if self.app.activeNode == self:
            self.app.activeNode = None

        if r.collidepoint(self.app.mouse_pos) and self.app.MANIPULATENODES:
            
            utilityHighlighted = False
            for x in self.inputs + self.outputs + self.utility:
                if x.checkIfUnderMouse():
                    utilityHighlighted = True
                    break

            if not utilityHighlighted:

                self.app.mouseAvailable = False
                self.app.activeNode = self
                w = 3

                self.moveKH.active = True

                if isinstance(self.STASHEDRESULT, np.ndarray) or self.godNode:
                    self.viewOutputKH.active = True
                self.disableKH.active = True
                self.deleteKH.active = True

                self.contextKH.active = True
                self.disconnectKH.active = True
                
                

                self.toolTip.render()

                if self.stashIm:
                    self.app.screen.blit(self.stashIm, self.pos + [0, self.height + 5])
                    t = self.app.fontSmaller.render("RENDER", True, [0,255,0])
                    self.app.screen.blit(t, self.pos + [2, self.height + 7])
                    pygame.draw.rect(self.app.screen, [0,255,0], [self.pos + [0, self.height + 5], self.stashIm.get_size()], width=1)


                if "del" in self.app.keypress:
                    self.delete()

                
        

            
            

        self.surface.fill([0,0,0,220])

        self.app.screen.blit(self.surface, self.pos)

        if self.errored:
            c = [255,0,0]
        elif self.usedStash:
            c = [0, 255,0]
        else:
            c = [255,255,255]

        pygame.draw.rect(self.app.screen, c, r, w)
        
        self.drawName(r, COLOR)

        if self.disabled:
            t = self.app.fontSmall.render("DISABLED", True, COLOR)
            self.app.screen.blit(t, self.pos + [0, 30])
        
        
        self.suggestions = []
        for x in self.inputs + self.outputs:
            x.render()

        

        if str(self.disp) != "ERROR":

            if isinstance(self.disp, np.ndarray):
                self.disp = str(self.disp.shape)

            if not isinstance(self.disp, str):
                self.disp = str(self.disp)

            t = self.app.fontSmaller.render(f"{self.disp[:10]}...", True, [100,100,100] if not self.godNode else [255,255,255])
            self.app.screen.blit(t, self.pos + [10,20])


        utilitySelectable = True
        utUse = []
        for x in self.utility:
            #x.render(utilitySelectable)
            if x.selected:
                utilitySelectable = False
            
            utUse.append(utilitySelectable)

        for x in range(len(self.utility)):
            ut = self.utility[-(x+1)]
            usable = utUse[-(x+1)]
            ut.render(usable)

    def nudgeStructure(self, fromNode: "Node" = None):
        
        if fromNode:
            self.realPos[0] = max(fromNode.realPos[0] + fromNode.WIDTH + 30, self.realPos[0])

        for x in self.outputs:
            l = x.fetchOutConnections()
            for y in l:
                y.parent.nudgeStructure(self)


    def clearCache(self):
        self.stashUtilityVal.clear()
        self.STASHEDRESULT = None
        self.stashIm = None

    
    def initCalc(self, force = False):
        if self.godNode or force:

            if not self.app.CALCING:
                self.app.CALCING = True
                self.nodeLog.add("\n\nCalcing...")
                start_new_thread(self.calc, ())

    def delete(self):
        
        if "del" in self.app.keypress:
            self.app.keypress.remove("del")

        for x in self.inputs + self.outputs:
            x.disconnect()

        if self in self.parentList:
            self.parentList.remove(self)

    def calc(self, usedStash=True):
        """
        Main calculation function for the Node.
        Refactored for better readability and maintainability.
        """
        INITIALSTASHSTATE = usedStash
        try:
            # Step 1: Validate inputs
            F_INPUTS, usedStash = self._validate_inputs(usedStash)
            if F_INPUTS is None:  # Error occurred during input validation
                return "ERROR", usedStash

            # Step 2: Check stash state
            usedStash = self._check_stash_state(usedStash)

            # Step 3: Perform calculation or use stash
            RESULT, usedStash = self._calculate_or_use_stash(F_INPUTS, usedStash)

            # Step 4: Process the result
            self._process_result(RESULT, usedStash)

            return RESULT, usedStash

        except Exception as e:
            self.nodeLog.add("FAIL CALCING", self.name)
            traceback.print_exc()
            self.errored = True
            self.disp = "ERROR"
            self.app.notify(f"Error in node calculation, {self.name}: {e}")
            if self.godNode:
                self.app.CALCING = False
            return ""

    def _validate_inputs(self, usedStash):
        """
        Validates the inputs for the Node and checks for errors.
        """
        F_INPUTS = []
        sendStash = usedStash

        for x in self.inputs:
            i, usedStashTemp = x.getValue(sendStash)

            if not usedStashTemp:
                usedStash = False

            if str(i) in {"ERROR", "None"}:
                self.errored = True
                if self.godNode:
                    self.app.CALCING = False
                return None, usedStash
            F_INPUTS.append(i)
        return F_INPUTS, usedStash

    def _check_stash_state(self, usedStash):
        """
        Checks whether the stash can be used based on utility values.
        """
        try:
            if len(self.stashUtilityVal) != len(self.utility):
                return False
            for i in range(len(self.utility)):
                if self.stashUtilityVal[i] != self.utility[i].value:
                    return False
        except Exception as e:
            self.nodeLog.add("EXCEPTION!!!", e)
            self.nodeLog.add(self.stashUtilityVal, self.utility)
            return False
        return usedStash

    def _calculate_or_use_stash(self, F_INPUTS, usedStash):
        """
        Performs the calculation or retrieves the result from the stash.
        """
        RESULT = None
        self.app.CALCNODE = self

        if self.disabled:
            for x in F_INPUTS:
                if isinstance(x, np.ndarray):
                    return x, usedStash
                
        
        self.stashUtilityVal = [x.value for x in self.utility]

        if usedStash and str(self.STASHEDRESULT) != "None" and not self.godNode:
            RESULT = self.STASHEDRESULT
        else:
            RESULT = self.f(*F_INPUTS)
            if self.hasMultipleOutputs:
                print("MULTIPLE OUTPUT CACL")
                print(type(RESULT))
            self.calcedThisExport = True
            self.STASHEDRESULT = RESULT

        return RESULT, usedStash

    def _process_result(self, RESULT, usedStash):
        """
        Processes the result of the calculation and updates the Node's state.
        """
        if isinstance(self.STASHEDRESULT, np.ndarray) and not usedStash:
            s = pygame.surfarray.make_surface(self.STASHEDRESULT).convert()
            self.stashIm = pygame.transform.scale_by(s, self.WIDTH / s.get_size()[0])

        if str(RESULT) != "None":
            self.disp = str(RESULT.shape) if isinstance(RESULT, np.ndarray) else str(RESULT)
        else:
            self.disp = ""

        self.errored = False
        if self.godNode:
            self.app.CALCING = False

        
        self.usedStash = usedStash


# Function to call and capture output
def track_function(func, *args, **kwargs):
    print(f"Calling {func.__name__} with arguments: {args} {kwargs}")
    result = func(*args, **kwargs)
    print(f"Output of {func.__name__}: {result}")
    return result

def getNodes(card, app):
    # Loop through all functions in the module

    NODES = {}

    for name, obj in inspect.getmembers(core.nodeFuncs):
        if inspect.isfunction(obj):

            nodeLog = Log()

            # Get function signature (inputs)
            signature = inspect.signature(obj)
            nodeLog.add(f"\n\nFunction name: {name}, Signature: {signature}")

            nodeLog.add(signature.parameters)
            print(signature.parameters)

            # Retrieve and check the docstring for the "DROPDOWN" marker
            docstring = obj.__doc__ if obj.__doc__ else "No docstring available"
            nodeLog.add(f"Docstring: {docstring}")
            needsUtility = []

            selections = []

            NODETYPE = "Misc"
            NODEHELP = ""
            nodeLog.add("Node type:")
            for x in docstring.split("\n"):
                nodeLog.add(x)
                if "MENUTYPE" in x:
                    NODETYPE = x.split(":")[1]

                elif "HELP" in x:
                    NODEHELP = x.split(":")[1]


                for y in utilityMap:
                    if utilityMap[y] in x:
                        needsUtility.append(y)

            nodeLog.add("Node type fetched:", NODETYPE)


            UTILITIES = []


            for x in docstring.split("\n"):

                for utilityType in needsUtility:

                    if utilityMap[utilityType] not in x:
                        continue

                    l = [utilityType]

                    for y in x.split(":"):
                        y = y.replace(" ", "")
                        if y and y != utilityMap[utilityType]:
                            l.append(y)
                            
                    UTILITIES.append(l)
                    break

            nodeLog.add(UTILITIES)
            

            inputs = {}
            for x in signature.parameters:
                print("INPUT:", x)
                inputs[x] = str(signature.parameters[x])

            output = signature.return_annotation
            print("OUTPUT RETURNED:", output)

            # Create the Node and pass the dropdown flag
            node = Node(app, card, name, inputs, output, obj, UTILITIES, NODETYPE, NODEHELP, nodeLog)
            #node.needs_dropdown = needs_dropdown  # Add a custom attribute to Node
            NODES[name] = node

            # Debugging output for dropdown detection
            if needsUtility:
                nodeLog.add(f"Function {name} requires utility.")

    return NODES
if __name__ == "__main__":
    getNodes()
