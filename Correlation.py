from matplotlib import pyplot
import numpy

class Correlation:


    def calculate_correaltion(self, data):
        names = list(data)
        correlations = data.corr()
        # plot correlation matrix
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = numpy.arange(0, len(data.columns), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        pyplot.show()
