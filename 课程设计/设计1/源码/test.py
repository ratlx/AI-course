from cProfile import label

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('car_classification.keras')

test_images=[]
test_directory = 'data/test'
img_size = (160, 160)
batch_size = 32
result_directory = 'result'

class_names = ['SUV', 'bus', 'family sedan', 'fire engine', 'heavy truck', 'jeep', 'minibus', 'racing car', 'taxi', 'truck']

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_directory,
    labels=None,
    shuffle=False,
    image_size=img_size,
    batch_size=batch_size
)

predictons = model.predict(test_dataset)

images = []
for batch in test_dataset:
    for img in batch:
        images.append(img.numpy())

for i in range(0, len(predictons), 25):
    plt.figure(figsize=(20, 20))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.imshow(images[i + j].astype("uint8"))
        plt.title(f"Predicted: {class_names[np.argmax(predictons[i + j])]}")
        plt.axis("off")
    plt.savefig(os.path.join(result_directory, f"result_{int(i/25+1)}.png"))
    plt.close()
