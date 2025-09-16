# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency..

## Neural Network Model

Include the neural network model diagram.
![425938676-6f1f98b2-a579-4e02-8883-70b62f6508cf](https://github.com/user-attachments/assets/cc79812b-528c-4183-8e35-fba40caf0a2a)

## DESIGN STEPS

### STEP 1:
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a Convolutional Neural Network (CNN).
### STEP 2:
Use the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of various clothing items.
### STEP 3:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers to extract features and classify clothing items.
### STEP 5:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs
### STEP 6:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7:
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: VARNIKA P
### Register Number: 212223240170
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Sudharsana Kumar S R')
        print('Register Number: 212223240162')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="402" height="249" alt="image" src="https://github.com/user-attachments/assets/0cc6b317-8b56-4cce-96b9-279111cf2e66" />

### Confusion Matrix

<img width="779" height="646" alt="image" src="https://github.com/user-attachments/assets/4d63500d-68d9-429e-8830-59fc928989dd" />

### Classification Report

<img width="528" height="341" alt="image" src="https://github.com/user-attachments/assets/5b6431a9-3709-4126-a9c6-aa388d1349b3" />


### New Sample Data Prediction

<img width="447" height="508" alt="image" src="https://github.com/user-attachments/assets/9e7fed1d-0216-4b64-ad7c-b6a211072b27" />

## RESULT

Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
