import pandas
import numpy as np
import matplotlib.pyplot as plt

# read in dataset
dataset = pandas.read_csv('./anonymized_dataset_preprocessed.csv')
print(dataset.head())

# for convenience split the dataset into two
dataset0 = dataset[dataset["target"]==0]
dataset1 = dataset[dataset["target"]==1]

# put into a list so we can iterate over it
datasets = [dataset, dataset0, dataset1]

print("")

# make a new pandas dataframe for the statistics
data_stats = pandas.DataFrame(columns=['ALL', 'NON-FRAUD', 'FRAUD'],
                              index=['transactions', 'accounts', 'merchants', 'cards', 'first names', 'last names',
                                     'emails', 'card countries', 'purchase countries', 'min amount', 'max amount'])

data_stats.loc['transactions'] = [d.shape[0] for d in datasets]
data_stats.loc['accounts'] = [len(d["AccountID"].unique()) for d in datasets]
data_stats.loc['merchants'] = [len(d["Merchant"].unique()) for d in datasets]
data_stats.loc['cards'] = [len(d["Card"].unique()) for d in datasets]
data_stats.loc['first names'] = [len(d["First Name"].unique()) for d in datasets]
data_stats.loc['last names'] = [len(d["Last Name"].unique()) for d in datasets]
data_stats.loc['emails'] = [len(d["Email"].unique()) for d in datasets]
data_stats.loc['card countries'] = [len(d["Country"].unique()) for d in datasets]
data_stats.loc['purchase countries'] = [len(d["GeoCode"].unique()) for d in datasets]
data_stats.loc['min amount'] = [min(d["Amount"]) for d in datasets]
data_stats.loc['max amount'] = [max(d["Amount"]) for d in datasets]

print(data_stats)

print("")

print("time span: ", min(dataset["date"]), " - ", max(dataset["date"]))
print("currencies: ", dataset["Currency"].unique())
print("num card issuer countries: ", len(dataset["Country"].unique()))
print("num purchase countries: ", len(dataset["GeoCode"].unique()))
print("amount range (in Euro): ", min(dataset["Amount"]), " - ", max(dataset["Amount"]))

print("")

print("num fraud card issuer countries: ", dataset1["Country"].unique())
print("num fraud purchase countries: ", dataset1["GeoCode"].unique())

# plt_width = 2
# plt_height = 1
# plt.figure()
#
# # amount
# plt.subplot(plt_height, plt_width, 1)
# plt.hist(dataset["Amount"])
#
# # num purchases per account
# plt.subplot(plt_height, plt_width, 2)
# plt.hist(dataset.groupby('AccountID').count()["Amount"], len(dataset["AccountID"].unique()))
# plt.ylabel('num transactions')
# plt.xlabel('customer')
#
# plt.show()
