import pandas as pd
df=pd.read_csv(r"C:\Users\31575\Desktop\Data.csv")
print(df.shape)
print(df.info())
print(df.describe())
df=df.drop(columns=['Name'])
df=df.drop(columns=['PassengerId'])
df=df.drop(columns=['Ticket'])
df['Exist_Cabin']=df['Cabin'].notna().astype(int)
df['Cabin_Class']=df['Cabin'].str[0].fillna('Unknown')
df['Cabin_Number']=df['Cabin'].str[1:]
df['Cabin_Number_Str']=df['Cabin_Number'].replace('', pd.NA)
df['Cabin_Number_a'] = pd.to_numeric(df['Cabin_Number_Str'].fillna(0), errors='coerce')
df['Cabin_Number'] = pd.to_numeric(df['Cabin_Number_a'].fillna(0), errors='coerce')
df=df.drop(columns=['Cabin','Cabin_Number_Str','Cabin_Number_a'])
print(df.info())
x=df.drop(columns=['Survived'])
y=df['Survived']
from sklearn.model_selection import train_test_split
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.4,random_state=42,stratify=y)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
mean_value=x_train['Age'].mean()
x_train['Age']=x_train['Age'].fillna(mean_value)
x_val['Age']=x_val['Age'].fillna(mean_value)
x_test['Age']=x_test['Age'].fillna(mean_value)
mode_value=x_train['Embarked'].mode()[0]
x_train['Embarked']=x_train['Embarked'].fillna(mode_value)
x_val['Embarked']=x_val['Embarked'].fillna(mode_value)
x_test['Embarked']=x_test['Embarked'].fillna(mode_value)
print(df.info())
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
categorical_features=['Cabin_Class','Pclass','Sex','Embarked']
numerical_features=['Age','SibSp','Parch','Fare','Cabin_Number']
preprocessor=ColumnTransformer(
    transformers=[
        ('numbers',StandardScaler(),numerical_features),
        ('categories',OneHotEncoder(),categorical_features)
    ]
)
x_preprocess_train=preprocessor.fit_transform(x_train)
x_preprocess_val=preprocessor.transform(x_val)
x_preprocess_test=preprocessor.transform(x_test)
import torch
x_preprocess_train_tensor=torch.from_numpy(x_preprocess_train).float()
x_preprocess_val_tensor=torch.from_numpy(x_preprocess_val).float()
x_preprocess_test_tensor=torch.from_numpy(x_preprocess_test).float()
y_train_tensor=torch.from_numpy(y_train.to_numpy()).long()
y_val_tensor=torch.from_numpy(y_val.to_numpy()).long()
y_test_tensor=torch.from_numpy(y_test.to_numpy()).long()
from torch.utils.data import DataLoader, TensorDataset
train_dataset=TensorDataset(x_preprocess_train_tensor,y_train_tensor)
val_dataset=TensorDataset(x_preprocess_val_tensor,y_val_tensor)
test_dataset=TensorDataset(x_preprocess_test_tensor,y_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(torch.isnan(x_preprocess_train_tensor).sum())
print(torch.isinf(x_preprocess_train_tensor).sum())
import torch.nn as nn
import torch.optim as optim
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2,output_size),
        )
    def forward(self, x):
        return self.layers(x)
model = MLP(input_size=22, hidden_size1=15, hidden_size2=15, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
num_epochs=36
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs,targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.3f}')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs,targets in val_dataloader:
        outputs = model(inputs)
        loss=criterion(outputs, targets)
        val_loss=loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct+= predicted.eq(targets).sum().item()
        print(f'Validation Loss:{val_loss/len(val_dataloader):.3f}')
        print(f'Validation Accuracy:{correct/total*100:.2f}%')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs,targets in test_dataloader:
        outputs = model(inputs)
        loss=criterion(outputs, targets)
        val_loss=loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct+= predicted.eq(targets).sum().item()
        print(f'Test Loss:{val_loss/len(test_dataloader):.3f}')
        print(f'Test Accuracy:{correct/total*100:.2f}%')




















