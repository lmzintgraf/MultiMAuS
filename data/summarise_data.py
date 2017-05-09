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
            % (card != purchase country) - (all) 0.1045
            % (card != purchase country) - (genuine) 0.1050
            % (card != purchase country) - (fraud) 0.0360
        As we can see, only 3% if fraudulentb transactions are made in countries which
        are not also the credit card country, and 10% for genuine transactoins. We believe
        that this is sufficiently small to be neglected. The results for data where we
        replace misisng values are:
            % (card != purchase country) - (all) : 0.0736
            % (card != purchase country) - (genuine) : 0.0740
            % (card != purchase country) - (fraud) : 0.0231
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt

# -----------------
# --- load data ---
# -----------------

# read in dataset
dataset01 = pandas.read_csv('./anonymized_dataset_preprocessed.csv')
dataset01.name = 'datasetAll'

# for convenience split the dataset into two
dataset0 = dataset01[dataset01["Target"] == 0]
dataset0.name = 'datasetGenuine'
dataset1 = dataset01[dataset01["Target"] == 1]
dataset1.name = 'datasetFraud'

# put into a list so we can iterate over it
datasets = [dataset01, dataset0, dataset1]

# ------------------------------
# --- basic analysis of data ---
# ------------------------------

print("")

print("time span: ", min(dataset01["Date"]), " - ", max(dataset01["Date"]))
print("currencies: ", dataset01["Currency"].unique())
print("num card issuer countries: ", len(dataset01["CardCountry"].unique()))
print("num purchase countries: ", len(dataset01["PurchaseCountry"].unique()))
print("amount range (in Euro): ", min(dataset01["Amount"]), " - ", max(dataset01["Amount"]))

print("")

print("% (card != purchase country) - (all): ", np.sum(dataset01["CardCountry"] != dataset01["PurchaseCountry"])/dataset01.shape[0])
print("% (card != purchase country) - (genuine): ", np.sum(dataset0["CardCountry"] != dataset0["PurchaseCountry"])/dataset0.shape[0])
print("% (card != purchase country) - (fraud): ", np.sum(dataset1["CardCountry"] != dataset1["PurchaseCountry"])/dataset1.shape[0])

print("")

# ----------------------------
# --- save aggregated data ---
# ----------------------------

# make a new pandas dataframe for the statistics
data_stats = pandas.DataFrame(columns=['ALL', 'NON-FRAUD', 'FRAUD'],
                              index=['transactions', 'accounts', 'merchants', 'cards', 'first names', 'last names',
                                     'emails', 'card countries', 'purchase countries', 'min amount', 'max amount'])

data_stats.loc['transactions'] = [d.shape[0] for d in datasets]
data_stats.loc['accounts'] = [len(d["AccountID"].unique()) for d in datasets]
data_stats.loc['merchants'] = [len(d["MerchantID"].unique()) for d in datasets]
data_stats.loc['cards'] = [len(d["CardID"].unique()) for d in datasets]
data_stats.loc['first names'] = [len(d["FirstName"].unique()) for d in datasets]
data_stats.loc['last names'] = [len(d["LastName"].unique()) for d in datasets]
data_stats.loc['emails'] = [len(d["Email"].unique()) for d in datasets]
data_stats.loc['card countries'] = [len(d["CardCountry"].unique()) for d in datasets]
data_stats.loc['purchase countries'] = [len(d["PurchaseCountry"].unique()) for d in datasets]
data_stats.loc['min amount'] = [min(d["Amount"]) for d in datasets]
data_stats.loc['max amount'] = [max(d["Amount"]) for d in datasets]

print(data_stats)

print("")

for d in datasets:

    agent_ids = d['AccountID'].unique()
    agent_stats = pandas.DataFrame(columns=range(len(agent_ids)))

    agent_stats.loc['genuine transactions'] = [sum(d.loc[d["AccountID"] == c, "Target"] == 0) for c in agent_ids]
    agent_stats.loc['fraudulent transactions'] = [sum(d.loc[d["AccountID"] == c, "Target"] == 1) for c in agent_ids]
    agent_stats.loc['merchants'] = [len(d.loc[d["AccountID"] == c, "MerchantID"].unique()) for c in agent_ids]
    agent_stats.loc['credit cards'] = [len(d.loc[d["AccountID"] == c, "CardID"].unique()) for c in agent_ids]
    agent_stats.loc['first names'] = [len(d.loc[d["AccountID"] == c, "FirstName"].unique()) for c in agent_ids]
    agent_stats.loc['last names'] = [len(d.loc[d["AccountID"] == c, "LastName"].unique()) for c in agent_ids]
    agent_stats.loc['emails'] = [len(d.loc[d["AccountID"] == c, "Email"].unique()) for c in agent_ids]
    agent_stats.loc['card countries'] = [len(d.loc[d["AccountID"] == c, "CardCountry"].unique()) for c in agent_ids]
    agent_stats.loc['purchase countries'] = [len(d.loc[d["AccountID"] == c, "CardCountry"].unique()) for c in agent_ids]
    agent_stats.loc['min amount'] = [min(d.loc[d["AccountID"] == c, "Amount"]) for c in agent_ids]
    agent_stats.loc['max amount'] = [max(d.loc[d["AccountID"] == c, "Amount"]) for c in agent_ids]

    agent_stats.to_csv('./custStats_{}.csv'.format(d.name), index_label=False)

for d in datasets:

    agent_ids = d['MerchantID'].unique()
    agent_stats = pandas.DataFrame(columns=range(len(agent_ids)))

    agent_stats.loc['genuine transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 0) for c in agent_ids]
    agent_stats.loc['fraudulent transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 1) for c in agent_ids]
    agent_stats.loc['customers'] = [len(d.loc[d["MerchantID"] == c, "AccountID"].unique()) for c in agent_ids]
    agent_stats.loc['credit cards'] = [len(d.loc[d["MerchantID"] == c, "CardID"].unique()) for c in agent_ids]
    agent_stats.loc['first names'] = [len(d.loc[d["MerchantID"] == c, "FirstName"].unique()) for c in agent_ids]
    agent_stats.loc['last names'] = [len(d.loc[d["MerchantID"] == c, "LastName"].unique()) for c in agent_ids]
    agent_stats.loc['emails'] = [len(d.loc[d["MerchantID"] == c, "Email"].unique()) for c in agent_ids]
    agent_stats.loc['card countries'] = [len(d.loc[d["MerchantID"] == c, "CardCountry"].unique()) for c in agent_ids]
    agent_stats.loc['purchase countries'] = [len(d.loc[d["MerchantID"] == c, "CardCountry"].unique()) for c in agent_ids]
    agent_stats.loc['min amount'] = [min(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]
    agent_stats.loc['max amount'] = [max(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]

    agent_stats.to_csv('./merchStats_{}.csv'.format(d.name), index_label=False)
