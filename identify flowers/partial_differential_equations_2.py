import torch
import numpy as np
import torch.nn as nn


device = torch.device( "cpu")
def formula(x,y):
    return torch.sin(np.pi*x/2)*torch.sin(np.pi*y*10)

#残差点
N_pde=20000
x_pde=2*torch.rand(N_pde, 1)
y_pde=2*torch.rand(N_pde, 1)
t_pde=torch.rand(N_pde, 1)
X_pde=torch.cat([x_pde, y_pde,t_pde], dim=1).float().to(device)
X_pde.requires_grad_(True)
#边界条件
N_bc=2000
x_bc_0=2*torch.zeros(N_bc, 1)
x_bc_2=2*torch.full((N_bc, 1),2)
x_bc=2*torch.rand(N_bc, 1)
y_bc_0=2*torch.zeros(N_bc, 1)
y_bc_2=2*torch.full((N_bc, 1),2)
y_bc=2*torch.rand(N_bc, 1)
t_bc=torch.rand(N_bc, 1)
X_0yt=torch.cat([x_bc_0,y_bc,t_bc],dim=1).float().to(device)
X_2yt=torch.cat([x_bc_2,y_bc,t_bc],dim=1).float().to(device)
X_x0t=torch.cat([x_bc,y_bc_0,t_bc],dim=1).float().to(device)
X_x2t=torch.cat([x_bc,y_bc_2,t_bc],dim=1).float().to(device)
U_bc_true=torch.zeros(N_bc, 1).float().to(device)
#初始条件
N_ic=2000
x_ic=2*torch.rand(N_ic, 1)
y_ic=2*torch.rand(N_ic, 1)
t_ic=torch.zeros(N_ic, 1)
X_ic=torch.cat([x_ic, y_ic, t_ic],dim=1).float().to(device)
U_ic_true=formula(x_ic,y_ic)
#建立模型
class PINN(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5):
        super(PINN,self).__init__()
        self.layer=nn.Sequential(
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
if __name__ == '__main__':
    device = torch.device("cpu")
    model = PINN(3,66,52,40,32,22)
    loss_mse = nn.MSELoss()
    def loss_pde(model,X_pde):
        u_pde=model(X_pde)
        grads=torch.autograd.grad(
            outputs=u_pde,
            inputs=X_pde,
            grad_outputs=torch.ones_like(u_pde),
            create_graph=True,
            retain_graph=True)[0]
        u_pde_x=grads[:,0].view(-1,1)
        u_pde_y=grads[:,1].view(-1,1)
        u_pde_t=grads[:,2].view(-1,1)
        u_pde_xx=torch.autograd.grad(
            outputs=u_pde_x,
            inputs=X_pde,
            grad_outputs=torch.ones_like(u_pde_x),
            create_graph=True,
            retain_graph=True
        )[0][:,0].view(-1,1)
        u_pde_yy=torch.autograd.grad(
            outputs=u_pde_y,
            inputs=X_pde,
            grad_outputs=torch.ones_like(u_pde_y),
            create_graph=True,
            retain_graph=True
        )[0][:,1].view(-1,1)
        f_x_y_t=988.5*torch.exp(-t_pde)*formula(x_pde,y_pde)-torch.exp(-2*t_pde)*torch.pow(formula(x_pde,y_pde),2)
        f=u_pde_t-(u_pde_yy+u_pde_xx+torch.pow(u_pde,2)+f_x_y_t)
        return loss_mse(f,torch.zeros_like(f))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs=20000
    w_pde=1.0
    w_bc=10.0
    w_ic=10.0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        U_ic_val=model(X_ic)
        loss_ic=loss_mse(U_ic_val,U_ic_true)

        U_bc_0yt_val=model(X_0yt)
        U_bc_2yt_val=model(X_2yt)
        U_bc_x0t_val=model(X_x0t)
        U_bc_x2t_val=model(X_x2t)
        loss_bc=loss_mse(U_bc_0yt_val,U_bc_true)+loss_mse(U_bc_2yt_val,U_bc_true)+loss_mse(U_bc_x0t_val,U_bc_true)+loss_mse(U_bc_x2t_val,U_bc_true)
        loss_pde_t=loss_pde(model,X_pde)
        loss_total=w_pde*loss_pde_t+w_bc*loss_bc+w_ic*loss_ic
        loss_total.backward()
        optimizer.step()
        if (epoch+1) % 1000 == 0:
            print(f'Epoch{epoch+1}/{num_epochs}|loss{loss_total.item():.4f}')







