# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="936" height="430" alt="image" src="https://github.com/user-attachments/assets/4812ffd1-c029-404b-baad-4735dc10a646" />


## DESIGN STEPS
### STEP 1: 
Import the required libraries such astorch, torch.nn, torch.optim, pandas, numpy, scikit-learn, matplotlib, and seaborn.

### STEP 2: 
Load the dataset from the CSV file (customers.csv), inspect its shape and columns, drop the ID column, and handle missing values.


### STEP 3: 
Preprocess the data and encode categorical columns using LabelEncoder, encode the target segment (A, B, C, D → 0, 1, 2, 3), scale numerical features using StandardScaler, and split into train/test sets (80/20).


### STEP 4: 
Convert the processed data into PyTorch tensors and wrap them in TensorDataset and DataLoader for mini-batch training.


### STEP 5: 
Define the PeopleClassifier neural network (input → 64 → 32 → 4), initialize CrossEntropyLoss and Adam optimizer, and train the model using train_model() for multiple epochs.


### STEP 6: 
Evaluate the trained model using a Confusion Matrix and Classification Report, then predict the customer segment for new sample data.


## PROGRAM

### Name: VD Natchathira

### Register Number: 212224230178

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)   # 4 classes: A, B, C, D

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="931" height="157" alt="image" src="https://github.com/user-attachments/assets/a53ca2a3-c151-4e3c-9043-f8c339206d60" />

### OUTPUT
## Confusion Matrix
<img width="757" height="635" alt="image" src="https://github.com/user-attachments/assets/8d511e06-06df-4e80-8d90-2f452e818329" />

## Classification Report
<img width="635" height="308" alt="image" src="https://github.com/user-attachments/assets/a6a413be-4d6a-455d-a3f9-a7326eb08a8b" />

### New Sample Data Prediction
<img width="1727" height="470" alt="image" src="https://github.com/user-attachments/assets/f7d47b15-1738-4298-87a5-1d9456ead95e" />

## RESULT
Hence, the Neural Network Classification Model was successfully developed and customer segments were predicted with good accuracy.
