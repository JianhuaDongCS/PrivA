import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from utility.helper import *
from utility.batch_test import *
from model import Model
from load_attribute import load_attribute
from gumbel_Softmax_layer import *
from fixed_matrix import *
import warnings
warnings.filterwarnings('ignore')
from time import time
import random

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(args)
    print("\n" + "[embed_size:  " + str(args.embed_size) + " ,dataset:  " + str(args.dataset) + "  lr: " + str(args.lr) + " ,batch_size:  " + str(args.batch_size) + " ,regs1:  " + str(args.regs1) + " ,regs2:  " + str(args.regs2)  + " ]" + "\n")
    args.device = torch.device('cuda:' +str(args.gpu_id))

    """
    *********************************************************
    Load user embedding.
    """
    Pre_BPR = torch.load(args.weights_path + args.dataset + "pre_train_e_u"  + '.pkl', map_location=args.device)
    u_embedding = Pre_BPR['U']
    i_embedding = Pre_BPR['I']
    print("u_embedding"+str(u_embedding))
    print("i_embedding"+str(i_embedding))

    Pre_BPR_Age = torch.load(args.weights_path + args.dataset + "pre_train_age"  + '.pkl', map_location=args.device)
    u_f_a_embedding = Pre_BPR_Age['U']
    i_f_a_embedding = Pre_BPR_Age['I']
    age_embedding = Pre_BPR_Age['age']
    NN_weight = Pre_BPR_Age['F.weight']
    NN_bias = Pre_BPR_Age['F.bias']
    print("u_f_a_embedding" + str(u_f_a_embedding))
    print("i_f_a_embedding" + str(i_f_a_embedding))
    print("age_embedding" + str(age_embedding))
    print("NN_weight" + str(NN_weight))
    print("NN_bias" + str(NN_bias))

    #Load user age(one hot).
    if (args.dataset == 'Book-Crossing/'):
        Age, Location = load_attribute(args)
        Attribute = [Age, Location]
    else:
        Age, Gender, Occupation = load_attribute(args)
        Attribute = [Age, Gender, Occupation]

    # print(Age[29,:])

    fixed_matrix = fixed_matrix_Age().to(args.device)


    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0x3f3f3f3f, 0


    model = Model(data_generator.n_users, data_generator.n_items, u_f_a_embedding, i_f_a_embedding, age_embedding, NN_weight, NN_bias, args).to(args.device)

    optimizer = optim.Adam(model.parameters(), weight_decay= args.weight_decay, lr=args.lr)
    All_loss, Discriminator_loss, Generator_loss = [], [], []


    for epoch in range(args.epoch):
        t1 = time()
        Loss_all, Loss_D, Loss_G= 0., 0., 0.
        #n_batch = data_generator.n_train // args.batch_size + 1
        n_batch = data_generator.n_users // args.batch_size + 1
        model.train()
        for idx in range(n_batch):

            users, _, _ = data_generator.sample()
            """
                Generator.
            """
            optimizer.zero_grad()
            G_MLP_input = u_embedding[users,:]
            G_MLP_output = model.Generator(G_MLP_input)
            age_set = gumbel_softmax(G_MLP_output,1,args)

            G_L1 = age_set
            G_L2 = torch.mm(age_set, fixed_matrix.T)

            valid_l1 = torch.ones([G_L1.size(0), G_L1.size(1)],device=args.device,requires_grad=False)
            valid_l2_l = torch.ones([G_L2.size(0), G_L2.size(1) - 1], device=args.device, requires_grad=False) * 2
            valid_l2_r = torch.ones([G_L2.size(0), 1], device=args.device, requires_grad=False)
            valid_l2 = torch.cat([valid_l2_l, valid_l2_r], dim = 1)

            G_loss = model.Generator_loss(G_L1, G_L2, valid_l1, valid_l2)

            """
                Discriminator.
            """

            age = Age[users,:]

            D_loss = model.Discriminator_loss(users, age, age_set)

            all_loss = model.Loss(G_loss, D_loss)


            all_loss.backward()

            optimizer.step()

            Loss_G += (G_loss/ n_batch)
            Loss_D += (D_loss/ n_batch)
            Loss_all += ((G_loss + D_loss).item())/ n_batch
        #break
        All_loss.append(Loss_all)
        Discriminator_loss.append(Loss_D.detach().cpu())
        Generator_loss.append(Loss_G.detach().cpu())
        if (epoch + 1) < 1:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: Loss == [%.5f] generator==[%.5f] discriminator==[%.5f] ' % (
                    epoch, time() - t1, Loss_all, Loss_G, Loss_D)
                print(perf_str)

            continue



        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs]: Loss == [%.5f] generator==[%.5f] discriminator==[%.5f]' % (
                epoch, time() - t1, Loss_all, Loss_G, Loss_D)
            print(perf_str)


        cur_best_pre_0, stopping_step, should_stop = early_stopping(Loss_all, cur_best_pre_0,
                                                                    stopping_step, expected_order='dec', flag_step=20)
        print(cur_best_pre_0, stopping_step, should_stop)
        result = (cur_best_pre_0, stopping_step)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            save_path = 'output/%sAge.result' % (args.dataset)
            ensureDir(save_path)
            f = open(save_path, 'a')

            f.write(
                'Age:  dataset:%s, attribute_dim:%d, lr:%.8f, c:%.8f, lambdal:%.8f, lambds:%.8f, weight_decay:%.8f, \n\t%s\n'
                % (args.dataset, args.attribute_dim, args.lr, args.c, args.lambdal, args.lambdas, args.weight_decay, result))
            f.close()

            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if Loss_all == cur_best_pre_0 and args.save_flag == 1:
            print(123)
            model.eval()
            Ageset = model.Generator(u_embedding)
            Ageset = gumbel_softmax(Ageset, 1, args)
            #torch.save(Ageset, args.ageset_path + args.dataset + str(args.lambdal)+"_"+str(args.lambdas) + "Age_G" + '.pkl')
            #torch.save(Ageset,args.ageset_path + args.dataset + "search_Age_G" + '.pkl')
            #torch.save(Ageset, args.ageset_path + args.dataset + str(args.c) + "Age_G" + '.pkl')
            torch.save(Ageset, args.ageset_path + args.dataset + "Age_G" + '.pkl')
            print('save the weights in path: ', args.ageset_path + args.dataset + "Age_G" + '.pkl')



