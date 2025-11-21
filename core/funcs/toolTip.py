
def splitText(text, maxLength=50):
    """
    Split text into lines no longer than maxLength.
    Explicit '\n' in the text is treated as a hard split with priority.
    """
    segments = text.split("\n")
    result = []

    for seg in segments:
        words = seg.split(" ")
        curr = ""
        for word in words:
            if len(curr) + len(word) <= maxLength:
                curr += word + " "
            else:
                if curr:
                    result.append(curr.rstrip())
                curr = word + " "
        if curr:
            result.append(curr.rstrip())

    return result



import pygame
class toolTip:
    def __init__(self, app, parent, title, text, titleColor = [255,255,255], textColor = [255,255,255]):
        self.app = app
        self.parent = parent
        self.title = title
        self.setText(text)

        self.invokeTick = self.app.GT(30, oneshot = True)
        self.app.tooltips.append(self)
        self.renderedLastTick = False
        
        self.titleColor = titleColor
        self.textColor = textColor
        self.reRender()

    def setText(self, text):
        self.text = splitText(text, maxLength=30)
        for i in range(len(self.text)):
            self.text[i] = self.text[i].strip(" ")

    def reRender(self):
        
        self.textRendered = []
        title = self.app.font.render(self.title, True, self.titleColor)

        maxX = title.get_size()[0] + 10
        maxY = 30
        for x in self.text:
            t = self.app.fontSmall.render(x, True, self.textColor)
            self.textRendered.append(t)
            maxX = max(maxX, t.get_size()[0])
            maxY += 25

        self.surf = pygame.Surface((maxX + 10, maxY + 10), pygame.SRCALPHA).convert_alpha()
        self.surf.fill((0,0,0,200))

        
        self.surf.blit(title, (5,5))
        yPos = 35
        for x in self.textRendered:
            self.surf.blit(x, (5,yPos))
            yPos += 25
        self.surfs = []

        for x in range(10):
            y = self.surf.copy()
            y.set_alpha(255 * x/9)

            self.surfs.append(y)
        self.surfs.append(self.surf)

    
    def render(self):
        self.renderedLastTick = True
        self.invokeTick.tick()
        if self.invokeTick.value >= self.invokeTick.max_value - 10:
            i = 10 - self.invokeTick.max_value + self.invokeTick.value
            self.app.TOOLTIP = self.surfs[i]
        



if __name__ == "__main__":
    toolTip(None, None, "Test")