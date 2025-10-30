import pygame


class ContextMenu:
    def __init__(self, parent, text, function=None):
        self.parent = parent
        self.app = parent.app
        self.text = text
        self.position = None
        self.function = function

        self.textSurf = self.app.fontSmaller.render(self.text, True, (255, 255, 255))

        # Check if parent has the contextMenus attribute
        if not hasattr(self.parent, 'contextMenus'):
            self.parent.contextMenus = []

        self.parent.contextMenus.append(self)


        self.menuIndex = self.parent.contextMenus.index(self)


    def toggleOn(self, position):
        """Toggle the context menu on."""
        if self not in self.app.contextMenuShow:
            self.app.contextMenuShow.append(self)
        self.position = [position[0], position[1] + 30]
        print(f"Context menu '{self.text}' toggled on at position {self.position}")

    def show(self):

        """Display the context menu."""

        maxx = 0
        for x in self.app.contextMenuShow:
            maxx = max(maxx, x.textSurf.get_width())

        pos = [self.position[0], self.position[1] + self.menuIndex*30]
        self.rect = pygame.Rect(pos[0], pos[1], maxx + 10, 30)

        if self.rect.collidepoint(self.app.mouse_pos):
            c = self.app.MAINCOLOR
            if "mouse0" in self.app.keypress:
                if self.function is not None:
                    self.function()
                self.app.contextMenuShow.clear()
                return
        


        else:
            c = [255, 255, 255]

        pygame.draw.rect(self.app.screen, (0, 0, 0), self.rect)
        pygame.draw.rect(self.app.screen, c, self.rect, 1)

        self.app.screen.blit(self.textSurf, [self.rect.x + 5, self.rect.y + 15 - self.textSurf.get_height() / 2])

        d = self.app.v2(self.rect.center).distance_to(self.app.v2(self.app.mouse_pos))
        self.FAR = d > maxx * 2
