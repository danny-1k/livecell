class Averager:
    def __init__(self, gamma:float=.9):
        self.i = 0
        self.value = 0
        self.gamma = gamma
        history = []

    def __iadd__(self, value:float):
        self.i +=1
        self.value = self.gamma*value + (1-self.gamma)*self.value
        self.history.append(self.value)