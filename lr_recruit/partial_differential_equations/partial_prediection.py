import torch
import pandas as pd
import numpy as np
import torch.nn as nn
device = torch.device('cuda:0')
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
    model_path= '/ml/identify flowers/pinn_train_weights.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"成功加载模型权重")

    csv_path='D:\download\submission.csv'
    df = pd.read_csv(csv_path)
    input=['x','y','t']
    x_np=np.array(df[input])
    x_tenser=torch.tensor(x_np).float().to(device)
    with torch.no_grad():
        u_pred = model(x_tenser)
        u_np=u_pred.cpu().numpy()
        df['u_pred']=u_np
        df.to_csv(csv_path,index=False)
        print("预测完成")




