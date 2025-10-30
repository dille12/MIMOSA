import pygame
import core.funcs.toolTip as toolTip
class Button:
    def __init__(self, app, text, pos, tooltip=None):
        self.text = text
        self.textR = app.font.render(self.text, True, [255,255,255])
        self.textRED = app.font.render(self.text, True, app.MAINCOLOR)
        self.textGREY = app.font.render(self.text, True, [155,155,155])
        self.app = app
        self.pos = app.v2(pos)
        self.rect = self.textR.get_rect()
        self.rect.inflate_ip(4,4)
        self.rect.topleft = self.pos
        self.locked = False
        if tooltip:
            self.toolTip = toolTip.toolTip(app, self, self.text, tooltip)
        else:
            self.toolTip = None

    def tick(self):
        highlight = False
        sel = False
        if self.rect.collidepoint(self.app.mouse_pos):
            highlight = True
            if not self.locked:
                sel = True



        if highlight and self.toolTip:
            self.toolTip.render()



        if self.locked:
            color = [155,155,155]
            t = self.textGREY

        elif sel:
            color = self.app.MAINCOLOR
            t = self.textRED
        else:
            color = [255,255,255]
            t = self.textR

        pygame.draw.rect(self.app.screen, color, self.rect, width=1)
        self.app.screen.blit(t, self.app.v2(self.rect.topleft) + [2,2])
        
        if sel and "mouse0" in self.app.keypress:
            return True
        return False