
import matplotlib.pyplot as plt
import numpy as np

a = [89.8, 95.8, 96, 94.8, 94.4, 93.3, 95.1, 95.4, 95.2]
labels = ['SVM+xgbfs', 'SVM+ufs', 'SVM+full', 'CART+xgbfs', 'CART+ufs', 'CART+full'
    ,'XGboost+xgbfs', 'XGboost+ufs', 'XGboost+full']

# this is for plotting purpose
index = np.arange(len(labels))
print(plt.style.available)
plt.style.use('ggplot')
plt.bar(index, a, color='blue')

plt.ylim(88, 97)
plt.xlabel('Model')
plt.ylabel('Percentage')
plt.xticks(index, labels, fontsize=10, rotation=30)
plt.title('F Measure Plot For Each Model')
plt.show()