"""
Gather some aggregated statistics about the data.

Fields we will use:
    MerchantID, Currency, CardID, Amount (in Euro), Date, Target, Country

"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from scipy.stats import beta
from datetime import datetime

# -----------------
# --- load data ---
# -----------------

# read in dataset
dataset01 = pandas.read_csv('./anonymized_dataset_preprocessed.csv')
# make the "date" column actually dates
dataset01["Date"] = pandas.to_datetime(dataset01["Date"])

# for convenience split the dataset into non-fraud(0)/fraud(1)
dataset0 = dataset01[dataset01["Target"] == 0]
# del dataset0["Target"]
dataset1 = dataset01[dataset01["Target"] == 1]
# del dataset1["Target"]

# give the datasets names
dataset01.name = 'datasetAll'
dataset0.name = 'datasetGenuine'
dataset1.name = 'datasetFraud'

# put datasets in list we can loop through
datasets = [dataset01, dataset0, dataset1]

# ------------------------------
# --- basic analysis of data ---
# ------------------------------

print("")
print("time span: ", min(dataset01["Date"]), " - ", max(dataset01["Date"]))
print("amount range (in Euro): ", min(dataset01["Amount"]), " - ", max(dataset01["Amount"]))
print("num non-fraudulent transactions: ", dataset0.shape[0])
print("num fraudulent transactions: ", dataset1.shape[0])
print("")

# plot transaction activity over time
plt.figure(figsize=(15, 5))
plt_idx = 1
for d in datasets:
    plt.subplot(1, 3, plt_idx)
    trans_dates = d["Date"].apply(lambda date: date.date())
    all_trans = trans_dates.value_counts(sort=False)
    date_num = matplotlib.dates.date2num(all_trans.index)
    plt.plot(date_num, all_trans.values, '.')
    plt_idx += 1
    plt.title(d.name)
    plt.xlabel('days (05.12.15 - 02.01.17)')
    plt.xticks([])
    if plt_idx == 2:
        plt.ylabel('amount')
plt.savefig('./aggregated/transactions_per_time')
plt.close()

# --------------------------------------
# --- in-depth analysis of customers ---
# --------------------------------------
# --> go through all the fields

# make a new pandas dataframe here we collect all statistics
data_stats_cols = ['ALL', 'NON-FRAUD', 'FRAUD']
data_stats = pandas.DataFrame(columns=data_stats_cols)


def get_transaction_prob(col_name, d01=dataset01, d0=dataset0, d1=dataset1):
    """ calculate fractions of transactions for given column """
    # generate
    num_trans = pandas.DataFrame(0, index=d01[col_name].value_counts().index, columns=data_stats_cols)
    num_trans['ALL'] = d01[col_name].value_counts()
    num_trans['NON-FRAUD'] = d0[col_name].value_counts()
    num_trans['FRAUD'] = d1[col_name].value_counts()
    num_trans = num_trans.fillna(0)
    num_trans /= np.sum(num_trans, axis=0)

    return num_trans


def get_grouped_prob(group_by, col_name):
    grouped_prob = dataset01.groupby([group_by, col_name]).size()
    grouped_prob = grouped_prob.groupby(level=0).apply(lambda x: x / sum(x))
    return grouped_prob


def get_transaction_dist(col_name):
    """ calculate fractions of transactions for given column """
    possible_vals = dataset01[col_name].value_counts().unique()
    trans_count = pandas.DataFrame(0, index=possible_vals, columns=data_stats_cols)
    trans_count['ALL'] = dataset01[col_name].value_counts().value_counts()
    trans_count['NON-FRAUD'] = dataset0[col_name].value_counts().value_counts()
    trans_count['FRAUD'] = dataset1[col_name].value_counts().value_counts()
    trans_count = trans_count.fillna(0)
    trans_count /= np.sum(trans_count.values, axis=0)

    # save
    trans_count.to_csv('./aggregated/{}_fract-dist.csv'.format(col_name), index_label=False)

    # print
    print(col_name)
    print(trans_count)
    print("")

    return trans_count


def plot_hist_num_transactions(trans_frac, col_name):
    """ method to plot histogram of number of transactions for a column """
    plt.figure(figsize=(10, 7))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.bar(range(trans_frac.shape[0]), trans_frac.values[:, i], label=trans_frac.index[i])
        plt.ylabel('num transactions')
        if i == 2:
            plt.xlabel(col_name)
    plt.savefig('./aggregated/{}_num-trans_hist'.format(col_name))
    plt.close()


def plot_bar_trans_prob(trans_frac, col_name, file_name=None):
    """ method to plot bar plot of number of transactions for a column """
    plt.figure()
    bottoms = np.vstack((np.zeros(3), np.cumsum(trans_frac, axis=0)))
    for i in range(trans_frac.shape[0]):
        plt.bar((0, 1, 2), trans_frac.values[i], label=trans_frac.index[i], bottom=bottoms[i])
    plt.xticks([0, 1, 2], ['all', 'non-fraud', 'fraud'])
    h = plt.ylabel('%')
    h.set_rotation(0)
    plt.title("{} Distribution".format(col_name))
    plt.legend()
    if not file_name:
        file_name = col_name
    plt.savefig('./aggregated/{}_num-trans_bar'.format(file_name))
    plt.close()

# --- CUSTOMER / FRAUDSTER STATISTICS ---

# -> regarding countries

# group countries by card
country_per_card01 = dataset01.groupby(['CardID', 'Country']).size().reset_index()
country_per_card0 = dataset0.groupby(['CardID', 'Country']).size().reset_index()
country_per_card1 = dataset1.groupby(['CardID', 'Country']).size().reset_index()

# get the probability of country per customer
country_per_card_prob = get_transaction_prob('Country', country_per_card01, country_per_card0, country_per_card1)

# save
country_per_card_prob.to_csv('./aggregated/country_per_customer_prob.csv', index_label=False)

# print
print("Country per customer prob: \n\n", country_per_card_prob.head(), "\n")

# --> regarding currencies, given the country

# group countries by card
currency_per_country01 = dataset01.groupby(['Country', 'Currency']).size().reset_index()
currency_per_country0 = dataset0.groupby(['Country', 'Currency']).size().reset_index()
currency_per_country1 = dataset1.groupby(['Country', 'Currency']).size().reset_index()

# get the probability of country per customer
num_trans = pandas.DataFrame(0, index=currency_per_country01['Currency'].value_counts().index, columns=data_stats_cols)
num_trans['ALL'] = currency_per_country01['Currency'].value_counts()
num_trans['NON-FRAUD'] = d0[col_name].value_counts()
num_trans['FRAUD'] = d1[col_name].value_counts()
num_trans = num_trans.fillna(0)
num_trans /= np.sum(num_trans, axis=0)

# save
country_prob.to_csv('./aggregated/country_currency_prob.csv', index_label=False)

# print
print("Currency per country prob: \n\n", country_prob.head(), "\n")

# --- CURRENCY ---

# how the merchant and the currency are tied together
plt.figure()
currencies = dataset01['Currency'].unique()
merchants = dataset01['MerchantID'].unique()
for curr_idx in range(len(currencies)):
    for merch_idx in range(len(merchants)):
        if currencies[curr_idx] in dataset01.loc[dataset01['MerchantID'] == merch_idx, 'Currency'].values:
            plt.plot(curr_idx, merch_idx, 'ko')
        plt.plot(range(len(currencies)), np.zeros(len(currencies))+merch_idx, 'r-', markersize=0.3)
plt.xticks(range(len(currencies)), currencies)
plt.savefig('./aggregated/merchant_vs_currency')
plt.close()

# --- AMOUNT ---

# # plot transaction activity over time
# plt.figure(figsize=(15, 12))
# plt_idx = 1
# print(dataset0["Country"].value_counts()[-5:])
# print(dataset1["Country"].value_counts()[-5:])
# for c in ['FR', 'NO', 'DE', 'DK', 'US', 'CA', 'GB']:
#     for d in datasets:
#         plt.subplot(7, 3, plt_idx)
#         # dates = d["Date"].apply(lambda date: date.datetime())
#         # dates = matplotlib.dates.date2num(dates)
#         amounts = d.loc[d['Country'] == c, "Amount"]
#         plt.plot(amounts.values, '.')
#         plt_idx += 1
#         plt.title(d.name)
#         plt.xlabel('days (05.12.15 - 02.01.17)')
#         plt.xticks([])
#         if plt_idx == 2:
#             plt.ylabel('num transactions')
# plt.savefig('./aggregated/amount_per_transaction_over_time')
# plt.close()



# for col in ['Country']:
#     for row in ['Currency']:
#     plt.figure(figsize=(15, 10))
#     d = dataset01
#     currencies = d[col].unique()
#     merchants = d[].unique()
#     for c_idx in range(len(currencies)):
#         for merch_idx in range(len(merchants)):
#             # print(currencies[c_idx])
#             # print(d.loc[d['MerchantID'] == merch_idx, 'Currency'])
#             if currencies[c_idx] in d.loc[d['Currency'] == merchants[merch_idx], col].values:
#                 plt.plot(c_idx, merch_idx, 'ko')
#             plt.plot(range(len(currencies)), np.zeros(len(currencies))+merch_idx, 'r-', markersize=0.3)
#     plt.xticks(range(len(currencies)), currencies)
#     plt.savefig('./aggregated/currency_vs_{}'.format(col))
#     plt.close()

for col in ['CardID', 'Amount']:
    plt.figure(figsize=(15, 5))
    plt_idx = 1
    for d in datasets:
        plt.subplot(1, 3, plt_idx)
        plt.plot(d[col], d["MerchantID"], '.')
        # plt.xticks(d[col], d[col].unique())
        plt_idx += 1
    plt.savefig('./aggregated/merchant_vs_{}'.format(col))

1/0

plt.figure()
bins = [0, 5, 15, 30, 50, 5000, 10600]
plt_idx = 1
for d in datasets:
    amount_counts, loc = np.histogram(d["Amount"], bins=bins)
    amount_counts = np.array(amount_counts, dtype=np.float)
    amount_counts /= np.sum(amount_counts)
    plt.subplot(1, 3, plt_idx)
    am_bot = 0
    for i in range(len(amount_counts)):
        plt.bar(plt_idx, amount_counts[i], bottom=am_bot, label='{}-{}'.format(bins[i], bins[i+1]))
        am_bot += amount_counts[i]
    plt_idx += 1
plt.legend()
plt.title("Amount distribution")
plt_idx += 1
plt.savefig('./aggregated/amount_distribution')
plt.close()

plt_idx = 1
for i in range(len(bins)-1):
    amounts = dataset0.loc[np.logical_and(bins[i] < dataset0["Amount"], dataset0["Amount"] < bins[i+1]), "Amount"]
    plt.subplot(1, len(bins)-1, plt_idx)
    plt.hist(amounts, bins=50)
    plt_idx += 1
plt.show()

# --- COUNTRIES ---

# # make some analysis only on france
# dataset01_france = dataset01.loc[dataset01["Country"] == 'FR']
# dataset0_france = dataset0.loc[dataset0["Country"] == 'FR']
# dataset1_france = dataset1.loc[dataset1["Country"] == 'FR']
# # plot transaction prob
# trans_prob_france = get_transaction_prob("Amount", dataset01_france, dataset0_france, dataset1_france)
# print(trans_prob_france.head())
# # plot_bar_trans_prob(trans_prob_france, 'Currency', file_name='currency_FR')
#
# # make some analysis only on the US
# dataset01_us = dataset01.loc[dataset01["Country"] == 'US']
# dataset0_us = dataset0.loc[dataset0["Country"] == 'US']
# dataset1_us = dataset1.loc[dataset1["Country"] == 'US']
# # plot transaction prob
# trans_prob_us = get_transaction_prob("Amount", dataset01_us, dataset0_us, dataset1_us)
# # plot_bar_trans_prob(trans_prob_us, 'Currency', file_name='currency_US')


# -- CURRENCIES ---

# get the transactions per currency
trans_prob_currency = get_transaction_prob("Currency")
# save
trans_prob_currency.to_csv('./aggregated/currency_trans_prob.csv', index_label=False)
# plot
plot_bar_trans_prob(trans_prob_currency, 'Currency')
# print
print("Currency prob: \n\n", trans_prob_currency.head(), "\n")

# check how many currencies there are per country
grouped_prob_country_currency = get_grouped_prob("Country", "Currency")
print("Currency prob per country: \n\n", grouped_prob_country_currency.head(), '\n')

# --- MERCHANTS ---

merch_trans_frac = get_transaction_prob("MerchantID")
plot_hist_num_transactions(merch_trans_frac, "MerchantID")

# for the four busiest merchants, plot the activity over time
num_busy_merchants = 4
busy_merchants = merch_trans_frac.index[:num_busy_merchants]
# plot transaction activity over time
plt.figure(figsize=(13, 7))
plt_idx = 1
min_date_num = matplotlib.dates.date2num(min(dataset01["Date"]))
max_date_num = matplotlib.dates.date2num(max(dataset01["Date"]))
for m in busy_merchants:
    for d in datasets:
        plt.subplot(num_busy_merchants, 3, plt_idx)
        dates = d.loc[d["MerchantID"] == m, "Date"].apply(lambda date: date.date())
        all_trans = dates.value_counts(sort=False)
        date_num = matplotlib.dates.date2num(all_trans.index)
        plt.plot(date_num, all_trans.values, '.')
        plt.xticks([])
        plt.xlim([min_date_num, max_date_num])
        if plt_idx < 4:
            plt.title(d.name)
        if plt_idx > (num_busy_merchants-1)*3:
            plt.xlabel('days (05.12.15 - 02.01.17)')
        if plt_idx % 3 == 1:
            plt.ylabel('merchant {}'.format(m))
        plt_idx += 1
plt.tight_layout()
plt.savefig('./aggregated/MerchantID_num-trans_time_top4')
plt.close()

# --- CREDIT CARDS ---

card_counts = get_transaction_dist("CardID")

# add to data_stats
data_stats = pandas.concat([data_stats, card_counts])

# plot
plt.figure()
bottoms = np.vstack((np.zeros(3), np.cumsum(card_counts, axis=0)))
for i in range(card_counts.shape[0]):
    plt.bar((0, 1, 2), card_counts.values[i], label=card_counts.index[i], bottom=bottoms[i])
plt.xticks([0, 1, 2], ['all', 'non-fraud', 'fraud'])
h = plt.ylabel('%')
h.set_rotation(0)
plt.title("Card Distribution")
plt.legend()
plt.savefig('./aggregated/card_distribution')
plt.close()

# # distribution of how many transactions are made by card
# transactions_per_card = dataset01["CardID"].value_counts().value_counts()
# print(transactions_per_card)
# a, b,  _, _ = beta.fit(transactions_per_card)
# plt.plot(transactions_per_card.index, transactions_per_card.values, '.')
# plt.plot(beta.pdf(np.linspace(1, 200, 200), a, b), 'r--')
# plt.xlabel('num transactions')
# plt.ylabel('num credit cards')
# plt.show()

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(dataset01["CardID"], dataset01["MerchantID"], '.', markersize=1)
plt.xlabel('card')
plt.ylabel('merchant')
plt.subplot(1, 3, 2)
plt.plot(dataset0["CardID"], dataset0["MerchantID"], '.', markersize=1)
plt.xlabel('card')
plt.subplot(1, 3, 3)
plt.plot(dataset1["CardID"], dataset1["MerchantID"], '.', markersize=1)
plt.xlabel('card')
plt.savefig('./aggregated/card_merchant_relation')

# --- DATE ---



# ----------------------
# --- aggregate data ---
# ----------------------

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
