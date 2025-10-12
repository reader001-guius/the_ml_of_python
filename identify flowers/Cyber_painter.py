import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
device = torch.device( "cpu")
#解释一下下：代码跟学长示范的有点出入，是因为我感觉学长示范的像是主程序中的写法，我是在前面先命名好函数，之后在main中安装学长的示范思路写
#定义特征提取工具
def get_features(image,model,layers):
    features={}
    x=image
    for name,layer in model.named_children():
        x=layer(x)
        if name in layers:
            features[layers[name]]=x
    return features
def gram_matrix(tensor):
    _,c,h,w=tensor.size()
    F=tensor.view(c,h*w)
    G=torch.mm(F,F.t())
    G.div_(h*w*c)
    return G
#定义TV_loss
def total_variation_loss(img):
    tv_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    tv_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    t_loss = tv_h + tv_w
    return t_loss
#定义反归一化
img_mean=torch.tensor([0.485, 0.456, 0.406],device=device).view(1, 3, 1, 1)
img_std=torch.tensor([0.229, 0.224, 0.225],device=device).view(1, 3, 1, 1)
def unnormalize(tensor):
    tensor=tensor*img_std+img_mean
    return tensor
if __name__=='__main__':
# 数据准备
    content_img = Image.open('content.png').convert('RGB')
    style_img = Image.open('style.png').convert('RGB')
    transform = torchvision.transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    content_img_tensor = transform(content_img).unsqueeze(0).to(device)
    style_img_tensor = transform(style_img).unsqueeze(0).to(device)
#初始化生成图
    gen_img_tensor=content_img_tensor.clone().requires_grad_(True)

# 加载预训练模型
    path=r"C:\Users\31575\.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth"
    vgg = models.vgg19(weights=None)
    state=torch.load(path,map_location=device)
    vgg.load_state_dict(state)
    vgg=vgg.features.eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
#调用特征提取工具
    content_layers={'21':'conv4_2'}
    style_layers={'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','28':'conv5_1'}
    all_layers={**content_layers,**style_layers}
   #[注意]：前面索引是计算机能够理解的VGG的具体层数；后面的名称是为了让程序更清晰，更好读
    content_features_dic=get_features(content_img_tensor,vgg,content_layers)
    style_features_dic=get_features(style_img_tensor,vgg,style_layers)
    content_features = content_features_dic['conv4_2']

    style_grams={}
    for layer_name,feature_tensor in style_features_dic.items():
        gram=gram_matrix(feature_tensor)
        style_grams[layer_name]=gram
#定义损失函数
    style_layer_weights={
        'conv1_1':0.40,
        'conv2_1':0.30,
        'conv3_1':0.20,
        'conv4_1':0.08,
        'conv5_1':0.02,
    }
    style_weight=1e4
    content_weight=1.0
    tv_weight=1e-6

    step=0
    def closure():
        optimizer.zero_grad(set_to_none=True)

        target_features=get_features(gen_img_tensor,vgg,all_layers)
        target_con_features=target_features['conv4_2']

        content_loss=F.mse_loss(target_con_features,content_features)

        style_loss=0.0
        for layer_name in style_grams.keys():
            weight=style_layer_weights[layer_name]
            target_gram=gram_matrix(target_features[layer_name])
            style_loss+=weight*F.mse_loss(target_gram,style_grams[layer_name].expand_as(target_gram))

        tv_loss=total_variation_loss(gen_img_tensor)

        loss=style_weight*style_loss+content_weight * content_loss + tv_weight * tv_loss
        loss.backward()

        global step
        step+=1
        if step%50==0:
            print(f"step{step:4d} |"
                  f"loss={loss.item():.4f} |"
                  f"content_loss={content_loss.item():.4f} |"
                  f"style_loss={style_loss.item():.4f} |"
                  f"tv_loss={tv_loss.item():.4f}"
                  )
        return loss
#选择优化器
    optimizer=torch.optim.LBFGS([gen_img_tensor],lr=1.0,max_iter=300)
 #训练循环
    optimizer.step(closure)
 #输出最终图像
    final_img=unnormalize(gen_img_tensor).clamp(0,1)
    torchvision.utils.save_image(final_img.cpu(), "output.png")
#因为是pytorch张量，就没有再转为PIL用save了



































