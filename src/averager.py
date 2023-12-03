class Averager:
    def __init__(self, gamma:float=.9) -> None:
        self.i = 0
        self.value = 0
        self.gamma = gamma
        self.history = []

    def __iadd__(self, value:float) -> None:
        self.i +=1
        self.value = self.gamma*value + (1-self.gamma)*self.value
        self.history.append(self.value)
        return self