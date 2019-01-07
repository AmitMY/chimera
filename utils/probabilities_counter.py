from collections import Counter


class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

        return self

    def compare(self, other):
        pmf = Pmf()
        for key1, prob1 in self.items():
            pmf[key1] += self[key1] - other[key1]
        return pmf

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))
