class FFNN:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.activations = []

    @property
    def activations(self):
        return self.activations
    @property
    def weights(self):
        return self.weights
    @property
    def biases(self):
        return self.biases

    @activations.setter
    def set_activations(self, nactive):
        self.activations = nactive
    @biases.setter
    def set_bases(self, nbiases):
        self.biases = nbiases
    @weights.setter
    def set_weights(self, nweights):
        self.weights = nweights
