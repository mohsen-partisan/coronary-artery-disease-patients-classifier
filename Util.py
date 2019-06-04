
import pandas as pd

class Util:

     def selected_features(self, data):
        non_zero = []
        i=0
        for item in data:
            if item !=0:
                non_zero.append(data.index(item))
                i = i+1
        return non_zero
