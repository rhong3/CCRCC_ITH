# Prepare labels for CCRCC ITH imaging project
import pandas as pd
import numpy as np

# # immune subtypes
# all = pd.read_csv("../cohort_DF.csv")
# all = all.loc[(all["Specimen_Type"] == "tumor_tissue") & (all["Tumor"] == "CCRCC")]
# samples = pd.read_csv("../CPTAC_ccRCC_ITH_meta_table_v1.0.tsv", sep="\t")
#
# all = all[["Patient_ID", "Slide_ID"]]
# samples = samples[["CASE_ID", "Slide_ID", "Immune_subtype_ITH_cohort", "Source"]]
#
# valid = pd.merge(all, samples, on=["Slide_ID"], how="inner")
# valid['label'] = valid["Immune_subtype_ITH_cohort"].str.split("im", n=1, expand=True)[1]
#
# valid['Slide_ID_tag'] = valid['Slide_ID'].str.split("-", expand=True)[2]
# valid = valid[["Patient_ID", "Slide_ID", "Slide_ID_tag", "label", "Source"]]
#
# valid = valid.dropna()
# valid.to_csv("../immune_label.csv", index=False, header=True)
# valid = pd.read_csv("../immune_label.csv")
# valid['label'] = valid['label'] - 1
# valid.to_csv("../immune_label.csv", index=False, header=True)
#
#
# # BAP1 mutation
# all = pd.read_csv("../cohort_DF.csv")
# all = all.loc[(all["Specimen_Type"] == "tumor_tissue") & (all["Tumor"] == "CCRCC")]
# samples = pd.read_csv("../CPTAC_ccRCC_ITH_meta_table_v1.0.tsv", sep="\t")
#
# all = all[["Patient_ID", "Slide_ID"]]
# samples = samples[["CASE_ID", "Slide_ID", "BAP1_mutant", "Source"]]
#
# valid = pd.merge(all, samples, on=["Slide_ID"], how="inner")
#
# valid['label'] = (~valid['BAP1_mutant'].isna()).astype(np.uint8)
#
# valid['Slide_ID_tag'] = valid['Slide_ID'].str.split("-", expand=True)[2]
# valid = valid[["Patient_ID", "Slide_ID", "Slide_ID_tag", "label", "Source"]]
#
# valid = valid.dropna()
# valid.to_csv("../BAP1_label.csv", index=False, header=True)


# Methylation subtypes
all = pd.read_csv("../cohort_DF.csv")
all = all.loc[(all["Specimen_Type"] == "tumor_tissue") & (all["Tumor"] == "CCRCC")]
samples = pd.read_csv("../CPTAC_ccRCC_combined_meta_table_v1.1.tsv", sep="\t")

all = all[["Patient_ID", "Slide_ID"]]
samples = samples[["CASE_ID", "Slide_ID", "Methylation_subtype_discovery110+confirmatory112_222", "Cohort"]]

valid = pd.merge(all, samples, on=["Slide_ID"], how="inner")

valid['Slide_ID_tag'] = valid['Slide_ID'].str.split("-", expand=True)[2]
valid.columns = ["Patient_ID", "Slide_ID", 'CASE_ID', "label", "Source", "Slide_ID_tag"]
valid = valid[["Patient_ID", "Slide_ID", "Slide_ID_tag", "label", "Source"]]
valid['label'] = valid['label']-1

valid = valid.dropna()
valid.to_csv("../methylation_label.csv", index=False, header=True)
