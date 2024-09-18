class LinearClassifier:
    def __init__(self, weight, learningrate):
        self.__weight = weight
        self.__learningrate = learningrate

    def query(self, input):
        return "гусениця" if input > self.__weight else "сонечко"

    def train(self, input, target):
        output = self.__weight * input
        error = target - output
        delta = self.__learningrate * (error / input)
        self.__weight += delta
