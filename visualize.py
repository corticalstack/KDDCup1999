import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualize:
    def __init__(self):
        pass

    def confusion_matrix(self, cm, model, target):
        title = 'Confusion Matrix - ' + model + ' - ' + target
        cm_array = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
        df = pd.DataFrame(cm_array, ['Positive', 'Negative'], ['Positive', 'Negative'])
        sns.heatmap(df, annot=True, annot_kws={'size': 16}, fmt='g', cbar=False, center=cm[1][0], cmap="Greys")
        plt.title(title)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        fig = plt.gcf()
        fig.savefig('viz/Graph - ' + title + '.png')
        plt.show()


