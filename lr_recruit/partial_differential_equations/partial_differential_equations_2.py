import torch
import torch.nn as nn


device = torch.device( "cuda:0")
def formula(x,y):
    return torch.sin(torch.pi*x/2)*torch.sin(torch.pi*y*10)

#残差点
N_pde=2000
x_pde=2*torch.rand(N_pde, 1).to(device)
y_pde=2*torch.rand(N_pde, 1).to(device)
t_pde=torch.rand(N_pde, 1).to(device)
X_pde=torch.cat([x_pde, y_pde,t_pde], dim=1).float().to(device)
X_pde.requires_grad_(True)

#边界条件
N_bc=2000
x_bc_0=torch.zeros(N_bc, 1)
x_bc_2=torch.full((N_bc, 1),2)
x_bc=2*torch.rand(N_bc, 1)
y_bc_0=torch.zeros(N_bc, 1)
y_bc_2=torch.full((N_bc, 1),2)
y_bc=2*torch.rand(N_bc, 1)
t_bc=torch.rand(N_bc, 1)
X_0yt=torch.cat([x_bc_0,y_bc,t_bc],dim=1).float().to(device)
X_2yt=torch.cat([x_bc_2,y_bc,t_bc],dim=1).float().to(device)
X_x0t=torch.cat([x_bc,y_bc_0,t_bc],dim=1).float().to(device)
X_x2t=torch.cat([x_bc,y_bc_2,t_bc],dim=1).float().to(device)
U_bc_true=torch.zeros(N_bc, 1).float().to(device)
#初始条件
N_ic=2000
x_ic=2*torch.rand(N_ic, 1).to(device)
y_ic=2*torch.rand(N_ic, 1).to(device)
t_ic=torch.zeros(N_ic, 1).to(device)
X_ic=torch.cat([x_ic, y_ic, t_ic],dim=1).float().to(device)
U_ic_true=formula(x_ic,y_ic).to(device)
#建立模型
class PINN(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5):
        super(PINN,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(input_size,hidden_size1),
            nn.Softplus(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.Softplus(),
            nn.Linear(hidden_size2,hidden_size3),
            nn.Softplus(),
            nn.Linear(hidden_size3,hidden_size4),
            nn.Softplus(),
            nn.Linear(hidden_size4,hidden_size5),
            nn.Softplus(),
            nn.Linear(hidden_size5,1)
        )
    def forward(self,x):
        return self.layer(x)
if __name__ == '__main__':
    model = PINN(3,166,166,155,88,66)
    model.to(device)
    loss_mse = nn.MSELoss()
    def loss_pde(model,X_pde,x_pde,y_pde,t_pde):
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
        f_x_y_t = 988.5 * torch.exp(-t_pde) * formula(x_pde, y_pde) - torch.exp(-2 * t_pde) * torch.pow(formula(x_pde, y_pde), 2)
        f=u_pde_t-(u_pde_yy+u_pde_xx+torch.pow(u_pde,2)+f_x_y_t)
        return loss_mse(f,torch.zeros_like(f)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0,max_iter=50,history_size=100)
    def closure():
        optimizer_lbfgs.zero_grad()
        U_ic_val = model(X_ic)

        U_bc_0yt_val = model(X_0yt)
        U_bc_2yt_val = model(X_2yt)
        U_bc_x0t_val = model(X_x0t)
        U_bc_x2t_val = model(X_x2t)
        Loss_PDE=loss_pde(model,X_pde,x_pde,y_pde,t_pde)
        Loss_IC=loss_mse(U_ic_val,U_ic_true)
        Loss_BC=loss_mse(U_bc_0yt_val,U_bc_true)+loss_mse(U_bc_2yt_val,U_bc_true)+loss_mse(U_bc_x0t_val,U_bc_true)+loss_mse(U_bc_x2t_val,U_bc_true)
        Loss_TOTAL=w_pde*Loss_PDE+w_ic*Loss_IC+w_bc*Loss_BC
        Loss_TOTAL.backward(retain_graph=True)
        return Loss_TOTAL

    num_epochs=20000
    num_lfbgs=2000
    w_pde=0.005
    w_bc=10.0
    w_ic=10.0
    W_PDE=10.0
    W_BC=1.0
    W_IC=1.0


    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        U_ic_val=model(X_ic).to(device)
        loss_ic=loss_mse(U_ic_val,U_ic_true)

        U_bc_0yt_val=model(X_0yt)
        U_bc_2yt_val=model(X_2yt)
        U_bc_x0t_val=model(X_x0t)
        U_bc_x2t_val=model(X_x2t)
        loss_bc=loss_mse(U_bc_0yt_val,U_bc_true)+loss_mse(U_bc_2yt_val,U_bc_true)+loss_mse(U_bc_x0t_val,U_bc_true)+loss_mse(U_bc_x2t_val,U_bc_true)
        loss_pde_t=loss_pde(model,X_pde,x_pde, y_pde, t_pde)
        loss_total=w_pde*loss_pde_t+w_bc*loss_bc+w_ic*loss_ic
        loss_total.backward(retain_graph=True)
        optimizer.step()
        if(epoch+1)%1000==0:
            N_pde = 2000
            x_pde = 2 * torch.rand(N_pde, 1).to(device)
            y_pde = 2 * torch.rand(N_pde, 1).to(device)
            t_pde = torch.rand(N_pde, 1).to(device)
            X_pde = torch.cat([x_pde, y_pde, t_pde], dim=1).float().to(device)
            X_pde.requires_grad_(True)

            # 边界条件
            N_bc = 2000

            x_bc_2 = torch.full((N_bc, 1), 2)
            x_bc = 2 * torch.rand(N_bc, 1)

            y_bc_2 = torch.full((N_bc, 1), 2)
            y_bc = 2 * torch.rand(N_bc, 1)
            t_bc = torch.rand(N_bc, 1)
            X_0yt = torch.cat([x_bc_0, y_bc, t_bc], dim=1).float().to(device)
            X_2yt = torch.cat([x_bc_2, y_bc, t_bc], dim=1).float().to(device)
            X_x0t = torch.cat([x_bc, y_bc_0, t_bc], dim=1).float().to(device)
            X_x2t = torch.cat([x_bc, y_bc_2, t_bc], dim=1).float().to(device)

            # 初始条件
            N_ic = 2000
            x_ic = 2 * torch.rand(N_ic, 1).to(device)
            y_ic = 2 * torch.rand(N_ic, 1).to(device)

            X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1).float().to(device)
            U_ic_true = formula(x_ic, y_ic).to(device)

        if (epoch+1) % 1000 == 0:
            print(f'Adam Epoch{epoch+1}/{num_epochs} | loss{loss_total.item():.6f} | Loss_IC: {loss_ic.item():.6f} | Loss_BC: {loss_bc.item():.6f} | Loss_PDE: {loss_pde_t.item():.6f}')
        if(epoch+1)  == num_epochs:
            print("\nAdam is over,step to LBFGS.")
    global i
    for i in range(num_lfbgs):
        optimizer_lbfgs.step(closure)
        if(i+1) % 100 == 0:
            U_ic_val = model(X_ic).to(device)

            U_bc_0yt_val = model(X_0yt)
            U_bc_2yt_val = model(X_2yt)
            U_bc_x0t_val = model(X_x0t)
            U_bc_x2t_val = model(X_x2t)
            Loss_PDE = loss_pde(model, X_pde, x_pde, y_pde, t_pde)
            Loss_IC = loss_mse(U_ic_val, U_ic_true)
            Loss_BC = loss_mse(U_bc_0yt_val, U_bc_true) + loss_mse(U_bc_2yt_val, U_bc_true) + loss_mse(U_bc_x0t_val,
                                                                                                       U_bc_true) + loss_mse(
                U_bc_x2t_val, U_bc_true)
            Loss_TOTAL = W_PDE * Loss_PDE + W_IC * Loss_IC + W_BC * Loss_BC
            print(f'LBFGS Epoch:{i+1}/{num_lfbgs} | loss:{Loss_TOTAL.item():.6f} | Loss_PDE:{Loss_PDE.item():.6f} | Loss_IC:{Loss_IC.item():.6f} | Loss_BC:{Loss_BC.item():.6f}')

        model_path= 'pinn_train_weights.pth'
    try:
        torch.save(model.state_dict(), model_path)
        print('Model saved in path: {}'.format(model_path))
    except Exception as e:
        print(f"Error saving model weights: {e}")












