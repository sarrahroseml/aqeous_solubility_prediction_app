#Import modules
import numpy as np
import pandas as pd 
import seaborn as sn
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
!pip install lazypredict
import lazypredict 
from lazypredict.Supervised import LazyRegressor
from tqdm import tqdm
!pip install rdkit
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

#Load training & validation dataset
df = pd.read_csv("/content/Aqueous solubility_9945.csv")
print(df.shape)
df.head()

#Remove null values of SMILES
no = df.SMILES.isnull().sum()
print(f"There are {no} null SMILES values ")
df.dropna(inplace = True)
new_shape = df.shape
print (f"New shape of df is {new_shape}")

##Remove outliers (logS values) 

#Generate a histogram 1. We find that it generally follows a normal distribution with a left tail that we want to chop off
sn.set_theme()
sn.displot(data=df, x='logS', binwidth = 1)
#Generate boxplot to identify specific outliers
sn.boxplot(data=df, x = 'logS')
#Filter values to remove outliers from normal distribution. Histogram distribution is a lot closer to normal distribution.
df_new = df[df.logS.apply(lambda x:x> - 7.5  and x <1.7)]
sn.displot(data=df_new, x = 'logS', binwidth = 1, kde = True )
#New boxplot
sn.boxplot(data=df_new, x = 'logS')

##Remove duplicates 
def canonical_SMILES(smiles):
  # For each SMILES string (smi) in the input list (smiles), it applies the Chem.CanonSmiles() function from the RDKit library. 
  # Function converts the input SMILES string into its canonical form.
  canon_smi = [Chem.CanonSmiles(smi) for smi in smiles]
  #Returns new list of canonical SMILES strings
  return canon_smi

canon_smiles = canonical_SMILES(df_new.SMILES)
df_new['SMILES'] = canon_smiles

#Create a list of duplicate smiles
#df_new['SMILES'].duplicated() returns a Series of Boolean values (True or False) that has the same index as df_new. 
#Each value is True if the corresponding 'SMILES' value in df_new is a duplicate, and False otherwise.
#df_new[df_new['SMILES'].duplicated()] uses this Boolean Series to index df_new, which selects only the rows where the Boolean Series has a True value. 
#Outputs a new DataFrame that only contains the rows of df_new where the 'SMILES' value is a duplicate.

duplicated_smiles = df_new[df_new['SMILES'].duplicated()]['SMILES'].values
print(f"There are {len(duplicated_smiles)} duplicated smiles")

#Select and filter duplicate SMILES (for visualisation)
df_new[df_new['SMILES'].isin(duplicated_smiles)].sort_values(by=['SMILES'])

#Keep the first duplicate & drop the rest
#The subset parameter is used to specify the columns to consider when identifying duplicates. Rows are considered duplicates if they have the same 'SMILES' value.
df_cl = df_new.drop_duplicates(subset='SMILES', keep='first')


#Remove duplicates from test set
test_set = pd.read_csv("/content/Drug_Like_Solubility _100.csv")
canon_smiles = canonical_SMILES(test_set.SMILES)
test_set["SMILES"] = canon_smiles
duplicate_test_smiles = test_set[test_set['SMILES'].duplicated()]['SMILES']
print(len(duplicate_test_smiles))

#Remove SMILES in training set that also appear in test set
test_set_SMILES = test_set.SMILES.values

#df_cl['SMILES'].isin(test_set_SMILES): Fenerates a Boolean Series that is True for every row in df_cl where the 'SMILES' value is in test_set_SMILES.
#The ~ operator is a NOT operation in Python, which inverts the Boolean values. So ~df_cl['SMILES'].isin(test_set_SMILES) 
#Generates a Series that is True for every row where the 'SMILES' value is NOT in test_set_SMILES.
df_cl_final = df_cl[~df_cl['SMILES'].isin(test_set_SMILES)]

print(f"Compounds in training set present in testing set is {len(df_cl)-len(df_cl_final)}")

#Filter logS values of test set to be within training set distribution
test_set = test_set[test_set['LogS exp (mol/L)'].apply(lambda x:x >-7.5 and x <1.7)]



#Generate RDKit Descriptors
def RDKit_descriptors(smiles):
  #Converts each SMILES string into a RDKit Mol object. 
  mols = [Chem.MolFromSmiles(i) for i in smiles]
  
  #Creates a MolecularDescriptorCalculator (MDC) object, which is used to calculate molecular descriptors. 
  #It's initialized with all the descriptor names available in RDKit's Descriptors module.
  calc = MoleculeDescriptors.MolecularDescriptorCalculator((x[0] for x in Descriptors._descList))
  
  #Gets the names of all descriptors that the MDC object can calculate
  desc_names = calc.GetDescriptorNames()
  
  #Initializes an empty list to store the molecular descriptors for each molecule.
  Mol_descriptors = []

  for mol in tqdm(mols):
    #Adds hydrogen atoms to the molecule. Some molecular descriptors take into account the number of hydrogen atoms in the molecule, so it's important to include them.
    mol = Chem.AddHs(mol)
    #Calculates the descriptors for the molecule using th MDC object.
    descriptors = calc.CalcDescriptors(mol)
    #Appends the calculated descriptors to the list of molecular descriptors.
    Mol_descriptors.append(descriptors)

  return Mol_descriptors, desc_names

#Function Call
Mol_descriptors, desc_names = RDKit_descriptors(df_cl_final['SMILES'])

#Create a dataframe containing the 200 molecular descriptors calculated for each compound in the training/validation set
df_200 = pd.DataFrame(Mol_descriptors, columns = desc_names)
df_200.head()



#Split dataset into training & validation
X_train, X_valid, y_train, y_valid = train_test_split(df_200,df_cl_final.logS,test_size = 0.1, random_state = 42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_valid_sc = scaler.transform(X_valid)

#Run LazyRegressor over 42 ML Models to identify best-performing models
lregs = LazyRegressor (verbose=0, ignore_warnings = True, custom_metric = None, random_state = 42)
models, prediction_tests = lregs.fit(X_train_sc, X_valid_sc, y_train, y_valid)
prediction_tests[:3]

#Train model & output predictions on validation set
model = LGBMRegressor(n_estimators = 1150, max_depth = 26, learning_rate = 0.04, random_state = 42)
model.fit(X_train_sc, y_train)
y_preds = model.predict(X_valid_sc)

#Plot Regression Graph
def plot_data(actual, predicted, title):
  rmse = np.sqrt(mean_squared_error(actual,predicted))
  R2 = r2_score(actual, predicted)
  plt.figure(figsize=(8,6))
 
  sn.regplot(x=predicted , y=actual,line_kws={"lw":2,'ls':'--','color':'red',"alpha":0.7})
  plt.title(title, color='red')
  plt.xlabel('Predicted logS(mol/L)', color='blue')
  plt.ylabel('Experimental logS(mol/L)', color ='blue')
  plt.xlim(-8,1)
  
  plt.grid(alpha=0.3)
  R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))
  rmse = mpatches.Patch(label="RMSE={:04.2f}".format(rmse))
  plt.legend(handles=[R2, rmse])

#How Model Performs on Validation Data
sn.set_theme(style="whitegrid")
plot_data(y_valid,y_preds,"Validation data: 10% of training/valid set ")



#Generate Molecular Descriptors for Testing Set
Mol_descriptors_test,desc_names_test = RDKit_descriptors(test_set['SMILES'])
test_set_200 = pd.DataFrame(Mol_descriptors_test, columns = desc_names_test)
X_test_sc = scaler.transform(test_set_200)
y_test_preds = model.predict(X_test_sc)

sn.set_theme(style="whitegrid")
plot_data(test_set['LogS exp (mol/L)'], y_test_preds, 'Test-data: 98 drug-like molecules')

#Save Model and Standard Scaler
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model,f)

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)





