class GameTick:
    def __init__(self, max_value=30, oneshot=False, nonMutable = False, startMaxed = False):
        self.value = 0
        self.max_value = max_value
        if startMaxed:
            self.value = self.max_value
        self.oneshot = oneshot
        self.nonMutable = nonMutable

    def tick(self):
        if self.value < self.max_value:
            self.value += 1
        if self.value < self.max_value:
            return False
        else:
            if not self.oneshot:
                self.value = 0
            return True

    def rounded(self):
        return round(self.value)
    
    def isMaxed(self):
        return self.value > self.max_value
    
    def ratio(self):
        return self.value / self.max_value