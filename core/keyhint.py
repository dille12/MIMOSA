import pygame
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import App

class KeyHint:
    def __init__(self, app: "App", key, purpose):
        self.app = app
        self.active = False
        self.key = key
        self.purpose = purpose
        self.app.keyhints.append(self)

        font = app.fontSmall
        key_surf = font.render(str(self.key) + ": ", True, (255, 255, 255))
        purpose_surf = font.render(str(self.purpose), True, (255, 255, 255))

        padding = 3
        spacing = 8

        w = key_surf.get_width() + spacing + purpose_surf.get_width() + padding * 2
        h = max(key_surf.get_height(), purpose_surf.get_height()) + padding * 2

        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 128))

        bg.blit(key_surf, (padding, padding))
        bg.blit(purpose_surf, (padding + key_surf.get_width() + spacing, padding))

        self.surface = bg
