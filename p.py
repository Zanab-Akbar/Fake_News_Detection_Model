import pandas as pd


#Read the csv files
df_test = pd.read_csv("test.tsv", sep='\t')
df_valid = pd.read_csv("valid.tsv", sep='\t')
df_train = pd.read_csv("train.tsv", sep='\t')

#Print out the headers

total= len(df_test) + len(df_train) + len(df_valid)

print("Total: ",total)





