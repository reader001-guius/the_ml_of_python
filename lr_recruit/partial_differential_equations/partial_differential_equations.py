import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  TensorDataset
import matplotlib.pyplot as plt

points = torch.empty(3000, 1)
points.uniform_(-5, 5)
dataset = TensorDataset(points)
batch_size = 888
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
def formula(x):
    return torch.pow((x), 2)+1


class Models(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5):
        super(Models, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size,hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2,hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3,hidden_size4),
            nn.Tanh(),
            nn.Linear(hidden_size4,hidden_size5),
            nn.Tanh(),
            nn.Linear(hidden_size5,1)
        )
    def forward(self,x):
        return self.layer(x)
if __name__=='__main__':
    device = torch.device("cpu")
    num_epochs=6666
    num_epochs_lbfgs=200
    x_ini = torch.tensor([[0.0]]).to(device)
    model=Models(input_size=1,hidden_size1=100,hidden_size2=88,hidden_size3=66,hidden_size4=55,hidden_size5=44)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50,history_size=50)
    def closure():
        optimizer_lbfgs.zero_grad()
        x = points.to(device).requires_grad_(True)
        y = model(x)
        gradient = torch.autograd.grad(y, x,
                                       grad_outputs=torch.ones_like(y),
                                       retain_graph=True, create_graph=True)[0]
        loss_ODE = criterion(gradient, 2 * x)
        y_val_ini = model(x_ini)
        loss_IC = criterion(y_val_ini, formula(x_ini))
        loss_lbfgs = loss_IC + loss_ODE
        loss_lbfgs.backward()
        return loss_lbfgs
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
            loss=loss_ODE+loss_IC
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss+=loss.item()

        if((epoch+1)%100==0):
            print(f'Epoch{epoch+1}, Loss: {running_loss/len(dataloader):.3f}')
    for i in range(num_epochs_lbfgs):
        loss_lbfgs=optimizer_lbfgs.step(closure)
        print(f'Epoch {i+1}, Loss: {loss_lbfgs.item():.3f}')
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
