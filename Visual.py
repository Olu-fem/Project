import matplotlib.pyplot as plt 
import unet_model
import alexnet_model
import resnet_model
from main import X_test, y_test

# Visualizing U-Net results
preds = unet_model.predict(X_test)
plt.figure(figsize=(10, 10))

for i in range(5):
    plt.subplot(5, 2, i*2 + 1)
    plt.imshow(X_test[i].astype('uint8'))
    plt.title(f"True Label: {y_test[i]}")
    
    plt.subplot(5, 2, i*2 + 2)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title(f"Predicted")
plt.show()

#Visualizing for Resnet
preds = resnet_model.predict(X_test)
plt.figure(figsize=(10, 10))

for i in range(5):
    plt.subplot(5, 2, i*2 + 1)
    plt.imshow(X_test[i].astype('uint8'))
    plt.title(f"True Label: {y_test[i]}")
    
    plt.subplot(5, 2, i*2 + 2)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title(f"Predicted")
plt.show()


#Visualizing for alexnet
preds = alexnet_model.predict(X_test)
plt.figure(figsize=(10, 10))

for i in range(5):
    plt.subplot(5, 2, i*2 + 1)
    plt.imshow(X_test[i].astype('uint8'))
    plt.title(f"True Label: {y_test[i]}")
    
    plt.subplot(5, 2, i*2 + 2)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title(f"Predicted")
plt.show()
