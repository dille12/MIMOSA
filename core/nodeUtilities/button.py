import pygame
from _thread import start_new_thread
class Button:
    def __init__(self, parent, app, text, callback):
        self.app = app
        self.parent = parent
        self.text = text
        self.callback = callback
        self.width = parent.WIDTH - 10
        self.height = 20
        self.y = parent.height
        parent.height += self.height + 5
        self.hovered = False
        self.pressed = False
        self.selected = False
        self.value = None

    def checkIfUnderMouse(self):
        POS = self.parent.pos + [4, self.y]
        r = pygame.Rect([POS, [self.width, self.height]])
        return r.collidepoint(self.app.mouse_pos)

    def render(self, utilitySelectable=True):
        POS = self.parent.pos + [4, self.y]
        r = pygame.Rect([POS, [self.width, self.height]])

        self.hovered = r.collidepoint(self.app.mouse_pos)
        if self.hovered and self.app.MANIPULATENODES and utilitySelectable:
            self.app.mouseAvailable = False
            if "mouse0" in self.app.keypress and not self.pressed:
                self.pressed = True
                start_new_thread(self.app.callbacks[self.callback], (self.app, self))
                
        else:
            self.pressed = False

        color = self.app.MAINCOLOR if self.hovered else [50, 50, 50]
        pygame.draw.rect(self.app.screen, color, r)
        pygame.draw.rect(self.app.screen, [255, 255, 255], r, width=1)

        t = self.app.fontSmall.render(self.text, True, [255, 255, 255])
        self.app.screen.blit(t, POS + [2, 0])
