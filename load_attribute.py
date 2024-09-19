import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

def load_attribute(args):

    if(args.dataset == 'Book-Crossing/'):

        # load age(one_hot)
        age = pd.read_csv(args.data_path + args.dataset + 'age.txt', ' ', header=None)
        age = np.array(age).astype(np.int64)
        age = torch.tensor(age).to(args.device)

        Age = [F.one_hot(age[i, -1], num_classes=7) for i in range(age.shape[0])]
        Age = torch.cat(Age, axis=-1)
        Age = torch.as_tensor(Age)
        Age = Age.reshape(age.shape[0], -1).to(args.device)

        # load location(one_hot)
        location = pd.read_csv(args.data_path + args.dataset + 'Location.txt', ' ', header=None)
        location = np.array(location).astype(np.int64)
        location = torch.tensor(location).to(args.device)

        Location = [F.one_hot(location[i, -1], num_classes=69) for i in range(location.shape[0])]
        Location = torch.cat(Location, axis=-1)
        Location = torch.as_tensor(Location)
        Location = Location.reshape(location.shape[0], -1).to(args.device)

        return Age, Location
    else:
        #load age(one_hot)
        age = pd.read_csv(args.data_path + args.dataset+'age.txt',' ', header=None)
        #age = pd.read_csv(args.data_path + args.dataset + 'bc_age_10.txt', ' ', header=None)
        age = np.array(age).astype(np.int64)
        age = torch.tensor(age).to(args.device)

        Age = [F.one_hot(age[i, -1], num_classes=7) for i in range(age.shape[0])]
        Age = torch.cat(Age, axis=-1)
        Age = torch.as_tensor(Age)
        Age = Age.reshape(age.shape[0], -1).to(args.device)
        #Age = Age[:, 1:].to(args.device)

        # load gender(one_hot)
        gender = pd.read_csv(args.data_path + args.dataset + 'gender.txt', ' ', header=None)
        gender = np.array(gender).astype(np.int64)
        gender = torch.tensor(gender).to(args.device)

        Gender = [F.one_hot(gender[i, -1], num_classes=2) for i in range(gender.shape[0])]
        Gender = torch.cat(Gender, axis=-1)
        Gender = torch.as_tensor(Gender)
        Gender = Gender.reshape(gender.shape[0], -1).to(args.device)

        # load occupation(one_hot)
        occupation = pd.read_csv(args.data_path + args.dataset + 'occupation.txt', ' ', header=None)
        occupation = np.array(occupation).astype(np.int64)
        occupation = torch.tensor(occupation).to(args.device)

        Occupation = [F.one_hot(occupation[i, -1], num_classes=21) for i in range(occupation.shape[0])]
        Occupation = torch.cat(Occupation, axis=-1)
        Occupation = torch.as_tensor(Occupation)
        Occupation = Occupation.reshape(occupation.shape[0], -1).to(args.device)



        return Age, Gender, Occupation
