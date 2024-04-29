# Set the backend to use mplcairo 
import matplotlib
import mplcairo
print('Default backend: ' + matplotlib.get_backend()) 
matplotlib.use("module://mplcairo.qt")
print('Backend is now ' + matplotlib.get_backend())

# IMPORTANT: Import these libraries only AFTER setting the backend
import matplotlib.pyplot as plt, numpy as np
from matplotlib.font_manager import FontProperties

from analyze_results import labels, scores
# Load Apple Color Emoji font 
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# Set up plot
# freqs = [301, 96, 53, 81, 42]
# labels = ['😊', '😱', '😂', '😄', '😛']
plt.figure(figsize=(20,8))
p1 = plt.bar(np.arange(len(labels)), scores, 0.8, color="lightblue")
plt.ylim(0, plt.ylim()[1]+30)

# Make labels
for rect1, label in zip(p1, labels):
    height = rect1.get_height()
    plt.annotate(
        label,
        (rect1.get_x() + rect1.get_width()/2, height+5),
        ha="center",
        va="bottom",
        fontsize=16,
        fontproperties=prop
    )

plt.show()