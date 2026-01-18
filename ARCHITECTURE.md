## Model Architecture

1. Input(8 tokens)：
```
# chem token([B, 600])
SMILES(string) + mpnn_encoder -> chem token

# morgan token([B, 1024]), maccs token([B, 167]), rdkit token([B, 210])
SMILES(string) + rdkit_encoder-> chem token, morgan token, maccs token, rdkit token

# comp token([B, 5])
Cationic_Lipid_to_mRNA_weight_ratio(float)
Cationic_Lipid_Mol_Ratio(float)
Phospholipid_Mol_Ratio(float)
Cholesterol_Mol_Ratio(float)
PEG_Lipid_Mol_Ratio(float)

# phys token([B, 12])
Purity_Pure(one-hot for Purity)
Purity_Crude(one-hot for Purity)
Mix_type_Microfluidic(one-hot for Mix_type)
Mix_type_Microfluidic(one-hot for Mix_type)
Cargo_type_mRNA(one-hot for Cargo_type)
Cargo_type_pDNA(one-hot for Cargo_type)
Cargo_type_siRNA(one-hot for Cargo_type)
Target_or_delivered_gene_FFL(one-hot for Target_or_delivered_gene)
Target_or_delivered_gene_Peptide_barcode(one-hot for Target_or_delivered_gene)
Target_or_delivered_gene_hEPO(one-hot for Target_or_delivered_gene)
Target_or_delivered_gene_FVII(one-hot for Target_or_delivered_gene)
Target_or_delivered_gene_GFP(one-hot for Target_or_delivered_gene)

# help token([B, 4])
Helper_lipid_ID_DOPE(one-hot for Helper_lipid_ID)
Helper_lipid_ID_DOTAP(one-hot for Helper_lipid_ID)
Helper_lipid_ID_DSPC(one-hot for Helper_lipid_ID)
Helper_lipid_ID_MDOA(one-hot for Helper_lipid_ID)

# exp token([B, 32])
Model_type_A549（one-hot for Model_type）
Model_type_BDMC（one-hot for Model_type）
Model_type_BMDM（one-hot for Model_type）
Model_type_HBEC_ALI（one-hot for Model_type）
Model_type_HEK293T（one-hot for Model_type）
Model_type_HeLa（one-hot for Model_type）
Model_type_IGROV1（one-hot for Model_type）
Model_type_Mouse（one-hot for Model_type）
Model_type_RAW264p7（one-hot for Model_type）
Delivery_target_dendritic_cell（one-hot for Delivery_target）
Delivery_target_generic_cell（one-hot for Delivery_target）
Delivery_target_liver（one-hot for Delivery_target）
Delivery_target_lung（one-hot for Delivery_target）
Delivery_target_lung_epithelium（one-hot for Delivery_target）
Delivery_target_macrophage（one-hot for Delivery_target）
Delivery_target_muscle（one-hot for Delivery_target）
Delivery_target_spleen（one-hot for Delivery_target）
Delivery_target_body（one-hot for Delivery_target）
Route_of_administration_in_vitro（one-hot for Route_of_administration）
Route_of_administration_intravenous（one-hot for Route_of_administration）
Route_of_administration_intramuscular（one-hot for Route_of_administration）
Route_of_administration_intratracheal（one-hot for Route_of_administration）
Sample_organization_type_individual（one-hot for Sample_organization_type）
Sample_organization_type_barcoded（one-hot for Sample_organization_type）
Value_name_log_luminescence(one-hot for Value_name)
Value_name_luminescence(one-hot for Value_name)
Value_name_FFL_silencing(one-hot for Value_name)
Value_name_Peptide_abundance(one-hot for Value_name)
Value_name_hEPO(one-hot for Value_name)
Value_name_FVII_silencing(one-hot for Value_name)
Value_name_GFP_delivery(one-hot for Value_name)
Value_name_Discretized_luminescence(one-hot for Value_name)
```

2. token projector

3. Channel split: to A[chem, Morgan, Maccs, rdkit] and B[comp, pays, help, exp]

4. Bi-directional cross attention

5. Token re-compose: back to [chem, Morgan, maccs, rdkit, comp, physical, help, exp]

6. Fusion layer(depends on user choice)

7. heads：
```
# regression head
size(float, training data already logged)

# classification head
toxic(boolean, 0/1)

# regression head
quantified_delivery(float, training data already z-scored)

# classification head
PDI_0_0to0_2(one-hot classification for PDI)
PDI_0_2to0_3(one-hot classification for PDI)
PDI_0_3to0_4(one-hot classification for PDI)
PDI_0_4to0_5(one-hot classification for PDI)

# classification head
Encapsulation_Efficiency_EE<50(one-hot classification for Encapsulation_Efficiency) 
Encapsulation_Efficiency_50<=EE<80(one-hot classification for Encapsulation_Efficiency)
Encapsulation_Efficiency_80<EE<=100(one-hot classification for Encapsulation_Efficiency)

# distribution head
Biodistribution_lymph_nodes(float, sum of Biodistribution is 1)
Biodistribution_heart(float, sum of Biodistribution is 1)
Biodistribution_liver(float, sum of Biodistribution is 1)
Biodistribution_spleen(float, sum of Biodistribution is 1)
Biodistribution_lung(float, sum of Biodistribution is 1)
Biodistribution_kidney(float, sum of Biodistribution is 1)
Biodistribution_muscle(float, sum of Biodistribution is 1)
```

