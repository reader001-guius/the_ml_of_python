import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

points = torch.empty(100, 1)
points.uniform_(-1, 1)
dataset = TensorDataset(points)
batch_size = 99
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
def formula(x):
    return torch.pow((x), 2)-1
class Models(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3):
        super(Models, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size,hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2,hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3,1)
        )
    def forward(self,x):
        return self.layer(x)
if __name__=='__main__':
    device = torch.device("cpu")
    num_epochs=666
    x_ini = torch.tensor([[1.0]]).to(device)
    model=Models(input_size=1,hidden_size1=39,hidden_size2=20,hidden_size3=15)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
    for epoch in range(num_epochs):
        model.train()

        running_loss=0.0
        for x in dataloader:
            optimizer.zero_grad()
            x = x[0].to(device).requires_grad_(True)
            y_val = model(x)
            gradient=torch.autograd.grad(y_val,x,
                                         grad_outputs=torch.ones_like(y_val),
                                         retain_graph=True)[0]
            loss_ODE=criterion(gradient,2*x)
            y_val_ini = model(x_ini)
            loss_IC = criterion(y_val_ini, formula(x_ini))
            loss=10*loss_ODE+loss_IC
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        print(f'epoch {epoch+1}, loss {running_loss/len(dataloader):.3f}')
    x_plot=points.detach().numpy()
    y_val_plot=model(points).detach().numpy()
    y_real_plot=formula(points).detach().numpy()
    plt.figure(figsize=(10,5))
    plt.scatter(x_plot[:,0],y_val_plot[:,0],c=y_val_plot,cmap='jet')
    plt.scatter(x_plot[:,0],y_real_plot[:,0],c=y_real_plot,cmap='jet')
    plt.title('PDE')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()





















