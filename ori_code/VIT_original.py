import numpy as np
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self,batch_size,image_size,patch_size,in_channels,embedding_dim,dropout):
        super(PatchEmbedding,self).__init__()

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        num_patches = (image_size//patch_size)**2
        self.num_patches = num_patches

        self.conv1 = nn.Conv2d(in_channels,out_channels=embedding_dim,kernel_size=patch_size,stride=patch_size)
        self.dropout = nn.Dropout(dropout)
        self.class_token = torch.randn(( batch_size,1,embedding_dim ))
        self.position_emd = torch.randn(( batch_size,num_patches+1,embedding_dim ))

    def forward(self,x):     # 输入维度(batch_size,in_channels=3,image_size=224,image_size=224)
        x = self.conv1(x)     # 结果维度(batch_size,embedding_dim=768,image_size//patch_size,image_size//patch_size)
        # x = x.flatten(2)
        x = x.view(self.batch_size, self.embedding_dim, self.num_patches)     # 结果维度(batch_size,embedding_dim=768,num_patches)
        x = x.transpose(1,2)     # 结果维度(batch_size,num_patches,embedding_dim=768)
        x = torch.concat((self.class_token, x), axis=1)     # 在维度num_patches拼接
        x = x + self.position_emd     # 结果维度(batch_size,197,embedding_dim=768)
        x = self.dropout(x)     # 此处的dropout运算是随机将x中的值用0替换, "丢弃神经元"
        return x


class Encoder(nn.Module):
    def __init__(self,batch_size,embedding_dim,num_heads,mlp_ratio,dropout,depth):
        super(Encoder, self).__init__()

        encoder_block_list = []
        for i in range(depth):
            encoder_block = EncoderLayer(batch_size,embedding_dim,num_heads,mlp_ratio,dropout)
            encoder_block_list.append(encoder_block)
        self.trans_encoder = nn.Sequential(*encoder_block_list)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self,x):    # 输入维度 (batch_size,196+1,embedding_dim)
        for trans_encoder_block in self.trans_encoder:
            x = trans_encoder_block(x)
        x = self.layer_norm(x)    # 结果维度 (batch_size,196+1,embedding_dim)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,batch_size,embedding_dim,num_heads,mlp_ratio,dropout):
        super(EncoderLayer, self).__init__()

        self.atten_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.atten = Attention(batch_size,embedding_dim,num_heads)

        self.dropout_atten = nn.Dropout(dropout)

        self.mlp_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.mlp = Mlp(embedding_dim,mlp_ratio,dropout)

    def forward(self,x):     # 输入维度 (batch_size,196+1,embedding_dim)
        residual = x
        x = self.atten_layer_norm(x)    # 结果维度 (batch_size,196+1,embedding_dim)
        x = self.atten(x)    # 结果维度 (batch_size,196+1,embedding_dim)

        x =  self.dropout_atten(x)

        x = x + residual

        residual = x
        x = self.mlp_layer_norm(x)    # 结果维度 (batch_size,196+1,embedding_dim)
        x = self.mlp(x)     # 结果维度 (batch_size,196+1,embedding_dim)

        x = x + residual

        return x


class Attention(nn.Module):
    def __init__(self,batch_size,embedding_dim,num_heads):
        super(Attention, self).__init__()

        self.qkv = embedding_dim // num_heads
        self.batch_size = batch_size
        self.num_heads = num_heads

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self,x):    # 输入维度 (batch_size,196+1,embedding_dim)
        # 输入都是x,只是W有区别
        Q = self.W_Q(x).view(self.batch_size, -1, self.num_heads, self.qkv).transpose(1, 2)    # 结果维度 (batch_size,num_heads,196+1,embedding_dim/num_heads)
        K = self.W_K(x).view(self.batch_size, -1, self.num_heads, self.qkv).transpose(1, 2)    # 结果维度 (batch_size,num_heads,196+1,embedding_dim/num_heads)
        V = self.W_V(x).view(self.batch_size, -1, self.num_heads, self.qkv).transpose(1, 2)    # 结果维度 (batch_size,num_heads,196+1,embedding_dim/num_heads)

        atten_result = Calculate_Attention()(Q, K, V, self.qkv)
        atten_result = atten_result.transpose(1,2).flatten(2)    # 结果维度 (batch_size,196+1,embedding_dim)

        return atten_result


class Calculate_Attention(nn.Module):
    def __init__(self):
        super(Calculate_Attention, self).__init__()

    def forward(self,Q,K,V,qkv):
        score = torch.matmul(Q,K.transpose(2,3)) / (np.sqrt(qkv))    # 结果维度 (batch_size,num_heads,196+1,196+1)
        score = nn.Softmax(dim=-1)(score)     # 在score的最后一个维度上归一化,即对同一batch、同一head的 QKT 的各行(第i行表示第i个token和其余token的相关性)做归一化
        score = torch.matmul(score,V)    # 结果维度 (batch_size,num_heads,196+1,embedding_dim/num_heads)
        return score


class Mlp(nn.Module):
    def __init__(self,embedding_dim,mlp_ratio,dropout):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embedding_dim,embedding_dim*mlp_ratio)
        self.activate1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_dim*mlp_ratio, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):    # 输入维度 (batch_size,196+1,embedding_dim)
        x = self.fc1(x)     # 结果维度 (batch_size,196+1,embedding_dim*mlp_ratio)
        x = self.activate1(x)
        x = self.dropout1(x)
        x = self.fc2(x)     # 结果维度 (batch_size,196+1,embedding_dim)
        x = self.dropout2(x)
        return x


class Classfication(nn.Module):
    def __init__(self,embedding_dim,num_classes,dropout):
        super(Classfication, self).__init__()
        self.fc1 = nn.Linear(embedding_dim,embedding_dim)
        self.fc2 = nn.Linear(embedding_dim,num_classes)
        self.relu = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):    # 输入维度 (batch_size,196+1,embedding_dim)
        x = x[:,0]        # 取batch_size维度的全部、"196+1"这个维度上的第一个、embedding_dim维度的全部, 即取class_token输入到分类器中进行最后的分类判别    结果维度 (batch_size,embedding_dim)
        x = self.fc1(x)    # 结果维度 (batch_size,embedding_dim)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)    # 结果维度 (batch_size,num_classes)
        x = self.dropout2(x)
        return x


class Vit(nn.Module):
    def __init__(self,
                 batch_size=1,         # 样本批处理个数
                 image_size=224,         # 图片大小
                 patch_size=16,         # patch大小
                 in_channels=3,         # 输入通道数
                 embedding_dim=768,         # 各token对应的特征维度
                 num_classes=1000,         # 类别数
                 depth=12,         # Encoder层的深度
                 num_heads=12,         # 多头自注意力的头数
                 mlp_ratio=4,         # 全连接层节点的倍数
                 dropout=0):         # dropout发生概率
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Vit, self).__init__()

        self.patch_embedding = PatchEmbedding(batch_size,image_size,patch_size,in_channels,embedding_dim,dropout)
        self.encoder = Encoder(batch_size,embedding_dim,num_heads,mlp_ratio,dropout,depth)
        self.classifier = Classfication(embedding_dim,num_classes,dropout)

    def forward(self,x):            # 输入维度(batch_size,in_channels=3,image_size=224,image_size=224)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)    # 结果维度 (batch_size,num_classes)

        return x



def main():
    batch_size = 3
    in_channels = 3
    image_size = 224

    vitmodel = Vit(batch_size=batch_size,         # 样本批处理个数
                 image_size=image_size,         # 图片大小
                 patch_size=16,         # patch大小
                 in_channels=in_channels,         # 输入通道数
                 embedding_dim=768,         # 各token对应的特征维度
                 num_classes=10,         # 类别数
                 depth=12,         # Encoder层的深度
                 num_heads=12,         # 多头自注意力的头数
                 mlp_ratio=4,         # 全连接层节点的倍数
                 dropout=0)

    input_img = torch.randn((batch_size, in_channels, image_size, image_size))
    out = vitmodel(input_img)
    print('结果维度:',out.shape)
    print('结果:',out)
    print('预测类别:',out.argmax(1))

if __name__ == '__main__':
    main()