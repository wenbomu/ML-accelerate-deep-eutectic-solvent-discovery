import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def Morgan_get():
    path = "HB"
    name = []
    mol = []
    infos = []
    morganFingerprint = []
    mols = pd.DataFrame()
    for root, dirs, list in os.walk(path):
        for i in list:
            info = {}
            name.append(i[:len(i) - 4])
            dir = os.path.join(root, i)
            mo = Chem.MolFromMolFile(dir)
            mol.append(mo)
            morganFingerprint.append(AllChem.GetMorganFingerprintAsBitVect(mo, 2,nBits=100, bitInfo=info))
            infos.append(info)
    mols["name"] = name
    mols["mol"] = mol
    mols["MorganFingerprint"] = morganFingerprint
    mols["infos"] = infos
    return mols

def data_get():
    mols = Morgan_get()
    data = pd.read_csv("use_data.csv")

    # # 数据归一化
    # feature_col = data.columns
    # for col in feature_col:
    #     if col != "water" and col != "HBA" and col != "HBD" and col != "ratio":
    #         data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

    def find_MorganFingerprint(chem):
        for i in range(0, len(mols)):
            if mols.loc[i, "name"] == chem:
                return mols.loc[i, "MorganFingerprint"]

    HBA_MorganFingerprint = []
    HBD_MorganFingerprint = []
    for i in range(0, len(data)):
        HBA_MorganFingerprint.append(find_MorganFingerprint(data.loc[i, "HBA"]))
        HBD_MorganFingerprint.append(find_MorganFingerprint(data.loc[i, "HBD"]))
    HBA_MorganFingerprint_vector = []
    HBD_MorganFingerprint_vector = []
    for i in range(0, len(HBA_MorganFingerprint)):
        s = HBA_MorganFingerprint[i].ToBitString()
        vectorA = [int(s[j]) for j in range(0, len(s))]
        HBA_MorganFingerprint_vector.append(vectorA)
    for i in range(0, len(HBD_MorganFingerprint)):
        s = HBD_MorganFingerprint[i].ToBitString()
        vectorB = [int(s[j]) for j in range(0, len(s))]
        HBD_MorganFingerprint_vector.append(vectorB)
    data["HBA_MorganFingerprint"] = HBA_MorganFingerprint_vector
    data["HBD_MorganFingerprint"] = HBD_MorganFingerprint_vector

    return data

data = data_get()
print(data)
# x = data.drop(["粘度（mPa·s）"],axis=1)
# # data = data.drop(['HBA_MorganFingerprint', 'HBD_MorganFingerprint', 'HBA_MorganCnt',
# #        'HBD_MorganCnt','HBA','HBD','water0','water1','water2'],axis=1)
# y = np.log(data["粘度（mPa·s）"])
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# data.to_csv("data.csv",index=False)