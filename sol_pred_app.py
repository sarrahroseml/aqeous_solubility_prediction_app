#Import relevant modules

from distutils.command.upload import upload
import numpy as np
import pandas as pd
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
import streamlit as st
from PIL import Image
import base64
import io

#Open saved trained lbgm regressoor model & standard scaler
with open('model_sol.pkl','rb') as f:
        model = pickle.load(f)
with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)

#200 Molecular Descriptors calculated for each molecule

#Physicochemical properties: Physical & chemical properties of the molecules. E.g.'MolWt' (molecular weight), 'HeavyAtomMolWt' (total weight of heavy atoms in mol), 'ExactMolWt' (exact mol weight), 'NumValenceElectrons' (number of valence e), and 'MolLogP' (log of partition coefficient between n-octanol and water).

#EState indices: Electrotopological state indices, which combine electronic (charge) information and topological (structure) information. They include 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', and 'MinAbsEStateIndex'.

#Fingerprint densities: 'FpDensityMorgan1', 'FpDensityMorgan2', and 'FpDensityMorgan3' are Morgan fingerprints of radius 1, 2, and 3, respectively. These are circular topological fingerprints.

#VSA descriptors: These are Volume of Molecular Surface Area descriptors, which are related to the molecular surface area portioned according to certain properties. They include 'PEOE_VSA' (partial equalization of orbital electronegativities volume), 'SMR_VSA' (Van der Waals surface area), and 'SlogP_VSA' (log P weighted surface area).

#Substructure counts: These descriptors count certain structural features or functional groups in the molecule. Examples include 'fr_Al_OH' (number of alcohol groups), 'fr_ether' (number of ether groups), 'fr_amide' (number of amide groups), and 'fr_nitro' (number of nitro groups).

#Ring descriptors: These descriptors provide information about the ring systems in the molecule. Examples include 'RingCount', 'NumAromaticRings', 'NumAliphaticRings', etc.

#Other descriptors: 'qed' stands for Quantitative Estimate of Drug-likeness, and 'TPSA' is the topological polar surface area.

descriptor_columns =  ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
       'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt',
       'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons',
       'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
       'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
       'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
       'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
       'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1',
       'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10',
       'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
       'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6',
       'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10',
       'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
       'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
       'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3',
       'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
       'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',
       'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
       'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
       'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
       'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
       'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
       'HeavyAtomCount', 'NHOHCount', 'NOCount',
       'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAromaticCarbocycles',
       'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
       'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
       'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
       'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
       'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
       'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
       'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
       'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
       'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
       'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
       'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
       'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
       'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
       'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
       'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
       'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
       'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
       'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
       'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
       'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
       'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
       'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

#Set text & colour themes
st.set_page_config(page_title='Aqueous Solubility Prediction App', layout='wide')

st.sidebar.markdown('<h2 style="color:#0D47A1;background-color:#B2EBF2;border-radius:10px;text-align:center"> Input compounds here for Solubility Prediction </h2>', unsafe_allow_html=True)

# Define solubility and explain why it is important
st.markdown('`Solubility` is defined as the maximum amount of solute that will dissolve in a given amount of solvent to form a saturated solution at a specified temperature, usually at room temperature. This Web App is developed by training 8,594 (90%) data points across 42 ML Models. The best model performance was obtained using the Light GBM Regressor(LGBMR). Below is a linear regression plot of the prediction of 98 drug-like compounds, against the ground truth values of their aqeous solubilities that were obtained via experimental validation.')

#Regression plot of 98 data points
#Regression plot of 98 data points
def plotting_reg_graph(df, title='Regression plot', xlabel='Predicted value', ylabel='Actual value'): 
    rsme_value = np.sqrt(mean_squared_error(df['Actual'],df['Predicted']))
    R2_value = r2_score(df['Actual'],df['Predicted'])
    
    rsme = mpatches.Patch(label="RMSE={:04.2F}".format(rsme_value))
    R2 = mpatches.Patch(label="R2={:04.2F}".format(R2_value))

    sn.regplot(x=df['Predicted'],y=df['Actual'],line_kws={"lw":2,'ls':'--','color':'red','alpha':0.7})
    plt.title(title, color='red')
    plt.xlabel(xlabel, color='blue')
    plt.ylabel(ylabel, color='blue')
    plt.xlim(-8,0.5)

    plt.grid(alpha=0.3)
   


#Test Data for Figure
test_set = pd.read_csv('test_98.csv')
plotting_reg_graph(test_set)

#Calculate 200 RDKit Descriptors
def RDKit_descriptors(smiles):
     #Converts each SMILES string into a RDKit Mol Object
     mols = [Chem.MolFromSmiles(i) for i in smiles]

     #Creates a MolecularDescriptorCalculator (MDC) object, used to calculate molecular descriptors
     #Initialized with all descriptor names available in RDKit's Descriptors module
     calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

     #Gets names of all descriptors that MDC can calculate
     des_names = calc.GetDescriptorNames()

     #Initialises empty list to store molecular descriptors for each mol
     Mol_descriptors = []
     for mol in mols: 
          #Adds H atoms to molecule. Some molecular descriptors take into account the number of hydrogen atoms in the molecule, so it's important to include them.
          mol = Chem.AddHs(mol)
          #Calculates descriptors for the molecule using MDC object
          descriptors = calc.CalcDescriptors(mol)
          #Appends the calculated descriptors to the list of molecular descriptors
          Mol_descriptors.append(descriptors)
     return Mol_descriptors, des_names

#Generate CSV file for output file to be downloaded
def file_download(data, file):
     df = data.to_csv(index=False)
     #CSV string df is first encoded to bytes using the default 'utf-8' codec, then those bytes are encoded into base64 format. finally, the base64 bytes are decoded back into a string.
     # The CSV data is in text format, but for it to be embedded into an HTML link (the href attribute specifically), it needs to be encoded. This is because HTML links are not designed to contain large amounts of text data like a CSV string. The encoding step converts the large CSV text into a compact, URL-friendly format.
     f = base64.b64encode(df.encode()).decode()

     #creates an HTML link that can be clicked to download the file. The href attribute holds the data in the form of a base64-encoded CSV file
     #When clicked, the download attribute tells the browser to download the file with the provided name.
     link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'

     return link 


#User input 
# Manual Input 
#Second argument is default value in text box
manual_SMILES = st.sidebar.text_input('Enter SMILE strings in single or double quotations seprated by comma:',"['CCCCO']")                                                   


#CSV input
st.sidebar.markdown('''or upload SMILE strings in CSV format, not that SMILE strings of molecules should be in 'SMILES' column: ''')          
upload_SMILES = st.sidebar.file_uploader("Click here to upload your CSV file")
st.sidebar.markdown("""**If you uploaded your CSV file, click below to get the solubility prediction**""")
prediction = st.sidebar.button("Predict LogS of molecules")

if manual_SMILES != "['CCCCO']":
     df = pd.DataFrame(eval(manual_SMILES), columns=['SMILES'])
     #Calculate 200 mol descriptors using RDkit_descriptors
     Mol_descriptors, desc_names = RDKit_descriptors(df['SMILES'])
     #Create dataframe 
     test_set_200 = pd.DataFrame(Mol_descriptors, columns=desc_names)
     #Using only the pre-selected descriptors listed above
     X_test = test_set_200[descriptor_columns]
     X_test_scaled = scaler.transform(X_test)
     X_logS = model.predict(X_test_scaled)
     predicted = pd.Dataq(X_logS,columns = ['Predicted LogS (mol/L)'])

     #Concatenate SMILES & predicted solubility
     output = pd.concat([df,predicted],axis=1)
     st.sidebar.markdown("""##See your output in the following table:""")
     
     #Display output in table & CSV link  
     st.sidebar.write(output)
     st.sidebar.markdown(file_download(output,"predicted_logS.csv"),unsafe_allow_html=True)

elif prediction:
     df2 = pd.read_csv(upload_SMILES)
     Mol_descriptors,desc_names = RDKit_descriptors(df2['SMILES'])
     test_set_200 = pd.DataFrame(Mol_descriptors,columns=desc_names)
     X_test = test_set_200[descriptor_columns]
     X_test_scaled = scaler.transform(X_test)
     X_logS = model.predict(X_test_scaled)
     predicted = pd.DataFrame(X_logS, columns=['logS (mol/L)'])
     output = pd.concat([df2['SMILES'],predicted],axis=1)
     st.sidebar.markdown('''##Your output is shown in the following table:''')
     st.sidebar.write(output)
     st.sidebar.markdown(file_download(output, "predicted_logS.csv"))
else: 
     st.markdown('''
<div style="border: 2px solid #ff6347; border-radius: 20px; padding: 3%; text-align:center; background-color: #282828; color: #ffffff;">
    <h3> style="color: #ff6347;"> Test this model now! </h2>
    <h6> You can use the sidebar to input your molecules. If you have a few molecules, simply put the SMILES in single or double quotations separated by a comma. If you have many molecules, upload them in a "SMILES" column and click the button that says "Predict logS of molecules". </h6>
    <h6 style="color: #ff6347; background-color: #ff6347; border-radius: 10px; padding: 3%; opacity: 0.9;"> Remember: predictions are more reliable if the compounds to be predicted are similar to those in the training dataset, with logS values ranging between -7.5 and 1.7.</h6>
</div>

''', unsafe_allow_html=True)


