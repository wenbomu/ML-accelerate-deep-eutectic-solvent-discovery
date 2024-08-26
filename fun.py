import pandas as pd
import os
from rdkit import Chem
from rdkit import DataStructs
import rdkit
from rdkit.Chem import AllChem
import torch
from torch.autograd import Variable
import numpy as np
import math

def get_mols():
    path = "HB"
    mols = pd.DataFrame()
    mol = []
    struct = []
    struct_info = []
    name = []
    morganFingerprint = []
    infos = []
    for root, dirs, list in os.walk(path):
        for i in list:
            info = {}
            name.append(i[:len(i)-4])
            dir = os.path.join(root,i)
            mo = Chem.MolFromMolFile(dir)
            mol.append(mo)
            morganFingerprint.append(AllChem.GetMorganFingerprintAsBitVect(mo,2,nBits=100,bitInfo=info))
            infos.append(info)
            info = {}
            struct.append(AllChem.GetMorganFingerprint(mo,2,bitInfo=info))
            struct_info.append(info)
            
    info = {}
    name.append("PEG")
    mo = Chem.MolFromSmiles("OCCO")
    mol.append(mo)
    morganFingerprint.append(AllChem.GetMorganFingerprintAsBitVect(mo,2,nBits=100,bitInfo=info))
    infos.append(info)
    info = {}
    struct.append(AllChem.GetMorganFingerprint(mo,2,bitInfo=info))
    struct_info.append(info)

    info = {}
    name.append("poly(ethylene_glycol)200")
    mo = Chem.MolFromSmiles("OCCO")
    mol.append(mo)
    morganFingerprint.append(AllChem.GetMorganFingerprintAsBitVect(mo,2,nBits=100,bitInfo=info))
    infos.append(info)
    info = {}
    struct.append(AllChem.GetMorganFingerprint(mo,2,bitInfo=info))
    struct_info.append(info)

    mols["name"] = name
    mols["MorganFingerprint"] = morganFingerprint
    mols["infos"] = infos

    return mols

def find(chem,mols):
    for i in range(0,len(mols)):
        if mols.loc[i,"name"] == chem:
            return mols.loc[i,"MorganFingerprint"]
        
def cal_similarity(vec1,vec2):
    sim = 0
    x = 0
    y = 0
    for i in range(len(vec1)):
        sim+=vec1[i]*vec2[i]
        x += vec1[i]*vec1[i]
        y += vec2[i]*vec2[i]
    return sim/(math.sqrt(x)*math.sqrt(y))
    # return sim/(x+y-sim)
    # return sim / max(x,y)
    # return sim*2 / (x+y)
    # return (x-sim)/y

def cal_similarity_Tanimoto(vec1,vec2):
    sim = 0
    x = 0
    y = 0
    for i in range(len(vec1)):
        sim+=vec1[i]*vec2[i]
        x += vec1[i]*vec1[i]
        y += vec2[i]*vec2[i]
    # return sim/(math.sqrt(x)*math.sqrt(y))
    return sim/(x+y-sim)

def filter():
    mols = get_mols()
    db = pd.read_csv("database.csv",encoding='utf-8')
    db = db.dropna(axis=0,how='all')
    db = db.dropna(axis=1,how='all')
    # print(db.columns)
    db = db.dropna(axis=0, how='any', subset=['pKa'])
    des = db[["MW g/mol（HBA）","MW g/mol（HBD）","water","DES","HBA:HBD比例",'pKa',"E+"]]
    # des = db[["DES", "HBA:HBD比例", 'pKa', "E+"]]
    des = des.drop_duplicates(keep='first')
    des = des.reset_index()
    des = des.drop(["index"],axis=1)

    des_HBA = []
    des_HBD = []
    for desi in des["DES"]:
        HBA_HBD = desi.split(':')
        des_HBA.append(HBA_HBD[0].strip())
        des_HBD.append(HBA_HBD[1].strip())
    des["HBA"] = des_HBA
    des["HBD"] = des_HBD

    H2O_MorganFingerprint = find('H2O',mols)
    s = H2O_MorganFingerprint.ToBitString()
    H2O = [int(s[j]) for j in range(0,len(s))]

    HBA_MorganFingerprint = []
    HBD_MorganFingerprint = []
    for i in range(0,len(des)):
        HBA_MorganFingerprint.append(find(des.loc[i,"HBA"],mols))
        HBD_MorganFingerprint.append(find(des.loc[i,"HBD"],mols))
        # print(find(des.loc[i,"HBD"],mols))
        # print(des.loc[i,"HBD"])
    
    HBA_MorganFingerprint_vector = []
    HBD_MorganFingerprint_vector = []
    for i in range(0,len(HBA_MorganFingerprint)):
        s = HBA_MorganFingerprint[i].ToBitString()
        vectorA = [int(s[j]) for j in range(0,len(s))]
        HBA_MorganFingerprint_vector.append(vectorA)
    for i in range(0,len(HBD_MorganFingerprint)):
        s = HBD_MorganFingerprint[i].ToBitString()
        vectorB = [int(s[j]) for j in range(0,len(s))]
        HBD_MorganFingerprint_vector.append(vectorB)

    MF = [] 
    for i in range(0,len(des)):
        ra = des.loc[i,"HBA:HBD比例"].split(":")
        h1 = [j*float(ra[0]) for j in HBA_MorganFingerprint_vector[i]]
        h2 = [j*float(ra[1]) for j in HBD_MorganFingerprint_vector[i]]
        h3 = []
        for j in range(len(h1)):
            h3.append(h1[j]+h2[j])
        if len(ra)== 3:
            for j in range(len(H2O)):
                h3[j]+=H2O[j]*float(ra[2])
        
        # h = []
        # for j in h3:
        #     if j>0:
        #         h.append(1)
        #     else:
        #         h.append(0)
        MF.append(h3)

    des["MF"] = MF

    return des

def get_des_sim(gen,condition,pre_des):
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    z = Variable(LongTensor(np.random.uniform(0, 10,  100)))
    gen_target = Variable(FloatTensor(condition))
    gen_vec = gen(z, gen_target)
    print(gen_vec)
    target_vec = []
    for i in gen_vec:
        if i>0.5:
            target_vec.append(1)
        else:
            target_vec.append(0)
    
    des = pre_des.copy()
    sim = []
    for i in range(len(des)):
        sim.append(cal_similarity(gen_vec.detach().cpu().numpy(),des.loc[i,"MF"]))
    des["sim"] = sim
    des.sort_values("sim",inplace=True,ascending=False)
    name = "_".join([str(x) for x in condition])
    # print(des)
    # des = des.reset_index()
    des.to_csv("sim3/"+name+"sim.csv",index=False)
    return des