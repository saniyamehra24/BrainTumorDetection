import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image=cv2.imread('C:\\Users\\hp\\OneDrive\\Desktop\\BrainTumor_Detection\\archive\\pred\\pred0.jpg')
img = Image.fromarray(image)
img_resized = img.resize((64, 64))
img_np = np.array(img_resized)
input_img = np.expand_dims(img_np, axis=0)

# Use predict to get probability scores for each class
predictions = model.predict(input_img)

# Extract the class with the highest probability
predicted_class = np.argmax(predictions)

print("Predicted Class:", predicted_class)
print("Probability Scores:", predictions)



