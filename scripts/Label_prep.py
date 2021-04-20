# Prepare labels for CCRCC ITH imaging project
import pandas as pd

all = pd.read_csv("../cohort_DF.csv")
all = all.loc[(all["Specimen_Type"] == "tumor_tissue") & (all["Tumor"] == "CCRCC")]
samples = pd.read_csv("../CPTAC_ccRCC_ITH_meta_table_v1.0.tsv", sep="\t")

all = all[["Patient_ID", "Slide_ID"]]
samples = samples[["CASE_ID", "Slide_ID", "Immune_subtype_ITH_cohort", "Source"]]

valid = pd.merge(all, samples, on=["Slide_ID"], how="inner")
valid['label'] = valid["Immune_subtype_ITH_cohort"].str.split("im", n=1, expand=True)[1]
valid['Slide_ID_tag'] = valid['Slide_ID'].str.split("-", expand=True)[2]
valid = valid[["Patient_ID", "Slide_ID", "Slide_ID_tag", "label", "Source"]]

valid.to_csv("../immune_label.csv", index=False, header=True)
