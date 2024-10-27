import alexnet_model
import resnet_model
import unet_model
from main import X_train, X_test, y_train, y_test


#Evaluate U-Net
unet_loss, unet_acc = unet_model.evaluate(X_test, y_test)
print(f"U-Net Accuracy: {unet_acc*100:.2f}%")

# Evaluate AlexNet
alexnet_loss, alexnet_acc = alexnet_model.evaluate(X_test, y_test)
print(f"AlexNet Accuracy: {alexnet_acc*100:.2f}%")

#Evaluate ResNet
resnet_loss, resnet_acc = resnet_model.evaluate(X_test, y_test)
print(f"ResNet Accuracy: {resnet_acc*100:.2f}%")
