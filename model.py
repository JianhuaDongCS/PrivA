import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, user_size, item_size, u_f_a_embedding, i_f_a_embedding, age_embedding,  NN_weight, NN_bias, args):
        super(Model, self).__init__()


        self.device = args.device
        self.dim = args.embed_size
        self.batch_size = args.batch_size
        self.p = 0.3
        self.lambda_l = args.lambdal
        self.lambda_s = args.lambdas

        self.U = u_f_a_embedding
        self.I = i_f_a_embedding
        self.age = age_embedding
        self.F = nn.Linear(self.dim, self.dim)
        self.F.weight = nn.Parameter(NN_weight , requires_grad=False)
        self.F.bias = nn.Parameter(NN_bias , requires_grad=False)

        self.decay1 = eval(args.regs1)[0]
        self.decay2 = eval(args.regs2)[0]


        self.attribute_dim = args.attribute_dim

        self.model = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            #nn.Dropout(self.p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            #nn.Dropout(self.p),

            nn.Linear(64, self.attribute_dim),
            nn.BatchNorm1d(self.attribute_dim),
            nn.Tanh(),
            #nn.Dropout(self.p),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                #print(m.weight)
                #print(m.bias)



    def Generator(self, x):

        return self.model(x)


    def Generator_loss(self, l1, l2, valid_l1, valid_l2):

        loss = torch.nn.MSELoss()
        out1 = loss(l1, valid_l1)
        out2 = loss(l2, valid_l2)
        out = self.lambda_l * out1 + self.lambda_s * out2
        return out / 100

    def Generator_loss1(self, l1, valid_l1):

        loss = torch.nn.MSELoss()
        out1 = loss(l1, valid_l1)
        out = out1
        return out / 100

    def  Discriminator_loss(self, users, age, age_set):

        number = age_set.sum(dim=1).detach()
        number = torch.repeat_interleave(number, self.dim, dim=0)
        number = number.reshape(self.batch_size, -1)

        loss = torch.nn.MSELoss()
        user_emb = self.U[users, :]

        user_age = torch.matmul(age.float(), self.age)

        u_ageset = torch.matmul(age_set.float(), self.age)
        user_ageset = (u_ageset / number).float()


        user_fusion_age_embedding = torch.add(user_emb, user_age)
        user_fusion_age_embedding = self.F(user_fusion_age_embedding)

        user_fusion_ageset_embedding = torch.add(user_emb, user_ageset)
        user_fusion_ageset_embedding = self.F(user_fusion_ageset_embedding)


        a_loss = loss(user_fusion_ageset_embedding, user_fusion_age_embedding)


        return a_loss

    def Loss(self, G_loss, D_loss):

        return G_loss +  D_loss

