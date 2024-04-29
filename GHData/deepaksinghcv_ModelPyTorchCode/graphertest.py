import matplotlib.pyplot as plt

number_of_training_images = [250,500,750,1000,1250,1500,1750,2000,2250,2500,2750]
exp_1_accuracy_values = [0.790333,0.82739,0.847402,0.849417,0.860702,0.874632,0.873692,0.881758,0.879404,0.882933,0.885476]

fig, ax = plt.subplots()

line1, = ax.plot(number_of_training_images, exp_1_accuracy_values , 
                marker='o', 
                color='b',
                label='baseline with 500 val images')

plt.xticks(number_of_training_images)
plt.xlabel("Number of training images")
plt.ylabel("Accuracy")
ax.legend()
plt.show()


