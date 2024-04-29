class DrawInputDataGraph():
    def __init__(self,df):
        self.df=df
        
    
    def drawGraph(self,plots,colors,labels):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
        fig.tight_layout()
        for i, ax in enumerate(axes.flat):
            for j in range(3):
                x = self.df.columns[plots[i][0]]
                y = self.df.columns[plots[i][1]]
                ax.scatter(self.df[self.df['target']==j][x], self.df[self.df['target']==j][y], color=colors[j])
                ax.set(xlabel=x, ylabel=y)

        fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
        plt.show()
        