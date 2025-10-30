import pygame

class Notification:
    def __init__(self, app, text):
        self.app = app
        self.text = text
        self.kp = []
        self.r = pygame.Rect(self.app.res/2, [0,0])
        self.r.inflate_ip(300, 200)

    def tick(self):

    
        pygame.draw.rect(self.app.screen, [0,0,0], self.r)
        pygame.draw.rect(self.app.screen, [255,255,255], self.r, width=1)
        t = self.app.font.render(self.text, True, [255,255,255])
        POS = self.app.v2(self.r.center) - [0,100] - self.app.v2(t.get_size())/2
        self.app.screen.blit(t, POS)

        if "space" in self.kp:
            return 2
        
        elif "esc" in self.kp:
            return 1
        
        else:
            return 0