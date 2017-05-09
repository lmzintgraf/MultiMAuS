"""
Gather some aggregated statistics about the data.

The original fields in the dataset are:
    AccountID, MerchantID, Currency, CardID, Amount (in Euro!), Date, Target,
    FirstName, LastName, Email, CardCountry, PurchaseCountry

Conclusions:
    -   Since the card issuer and purchase country are only very rarely not the same,
        we only include the "card country" (since "purchase country" had some missing data)
        in our simulator; we simply call this "country". (Note: I also checked that this
        is true when we don't replace "purchase country" with "card country" when values
        are missing. The results were:
            (card != purchase country) - (all) 10.45 %
            (card != purchase country) - (genuine) 10.50 %
            (card != purchase country) - (fraud) 3.60 %
        As we can see, only 3% if fraudulentb transactions are made in countries which
        are not also the credit card country, and 10% for genuine transactoins. We believe
        that this is sufficiently small to be neglected. The results for data where we
        replace misisng values are:
            (card != purchase country) - (all) : 7.36 %
            (card != purchase country) - (genuine) : 7.40 %
            (card != purchase country) - (fraud) : 2.31 %
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt

# -----------------
# --- load data ---
# -----------------

# read in dataset
dataset01 = pandas.read_csv('./anonymized_dataset_preprocessed.csv')

# for convenience split the dataset into two
dataset0 = dataset01[dataset01["Target"] == 0]
dataset1 = dataset01[dataset01["Target"] == 1]

# give the datasets names
dataset01.name = 'datasetAll'
dataset0.name = 'datasetGenuine'
dataset1.name = 'datasetFraud'

# ------------------------------
# --- basic analysis of data ---
# ------------------------------

print("")

# basic overview of all the transactions made
print("time span: ", min(dataset01["Date"]), " - ", max(dataset01["Date"]))
print("currencies: ", dataset01["Currency"].unique())
print("amount range (in Euro): ", min(dataset01["Amount"]), " - ", max(dataset01["Amount"]))
print("")

# count number of countries
print("num card issuer countries: ", len(dataset01["Country"].unique()))
print("num purchase countries: ", len(dataset01["PurchaseCountry"].unique()))
print("")

# analyse how often the card issuing country and purchase country differ
print("(card != purchase country) - (all): ", 100*round(np.sum(dataset01["Country"] != dataset01["PurchaseCountry"])/dataset01.shape[0], 4), '%')
print("(card != purchase country) - (genuine): ", 100*round(np.sum(dataset0["Country"] != dataset0["PurchaseCountry"])/dataset0.shape[0], 4), '%')
print("(card != purchase country) - (fraud): ", 100*round(np.sum(dataset1["Country"] != dataset1["PurchaseCountry"])/dataset1.shape[0], 4), '%')
print("")

# we are going to only keep the card country as a feature
# because the card and purchase country are mostly overlapping,
# and the purchase country data is incomplete
del dataset01["PurchaseCountry"]
del dataset0["PurchaseCountry"]
del dataset1["PurchaseCountry"]

# unique credit cards
print("num cards (all): ", dataset01["CardID"].unique().shape[0])
print("num cards (genuine): ", dataset0["CardID"].unique().shape[0])
print("num cards (fraud): ", dataset1["CardID"].unique().shape[0])
# unique names
print("num names (all): ", dataset01["Name"].unique().shape[0])
print("num names (genuine): ", dataset0["Name"].unique().shape[0])
print("num names (fraud): ", dataset1["Name"].unique().shape[0])
print("")

# since the number of credit cards and names is almost equal (esp for fraud)
# we are going to remove the "name" column and identify a customer only
# by their credit card
del dataset01["Name"]
del dataset0["Name"]
del dataset1["Name"]

# ----------------------
# --- aggregate data ---
# ----------------------

# make a new pandas dataframe for the statistics
data_stats = pandas.DataFrame(columns=['ALL', 'NON-FRAUD', 'FRAUD'])

datasets = [dataset01, dataset0, dataset1]

data_stats.loc['transactions'] = [d.shape[0] for d in datasets]
data_stats.loc['merchants'] = [len(d["MerchantID"].unique()) for d in datasets]

data_stats.loc['cards'] = [len(d["CardID"].unique()) for d in datasets]
data_stats.loc['cards, single use'] = [sum(d["CardID"].value_counts() == 1) for d in datasets]
data_stats.loc['cards, multi use'] = [sum(d["CardID"].value_counts() > 1) for d in datasets]

data_stats.loc['countries'] = [len(d["Country"].unique()) for d in datasets]
data_stats.loc['min amount'] = [min(d["Amount"]) for d in datasets]
data_stats.loc['max amount'] = [max(d["Amount"]) for d in datasets]
data_stats.loc['avg amount'] = [np.average(d["Amount"]) for d in datasets]
data_stats.loc['min trans/card'] = [min(d["CardID"].value_counts()) for d in datasets]
data_stats.loc['max trans/card'] = [max(d["CardID"].value_counts()) for d in datasets]
data_stats.loc['avg trans/card'] = [np.average(d["CardID"].value_counts()) for d in datasets]

print(data_stats)

print("")

for d in datasets:

    agent_ids = d['MerchantID'].unique()
    agent_stats = pandas.DataFrame(columns=range(len(agent_ids)))

    agent_stats.loc['genuine transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 0) for c in agent_ids]
    agent_stats.loc['fraudulent transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 1) for c in agent_ids]
    agent_stats.loc['credit cards'] = [len(d.loc[d["MerchantID"] == c, "CardID"].unique()) for c in agent_ids]
    agent_stats.loc['countries'] = [len(d.loc[d["MerchantID"] == c, "Country"].unique()) for c in agent_ids]
    agent_stats.loc['min amount'] = [min(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]
    agent_stats.loc['max amount'] = [max(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]

    print(agent_stats)
    print("")

    agent_stats.to_csv('./merchStats_{}.csv'.format(d.name), index_label=False)
