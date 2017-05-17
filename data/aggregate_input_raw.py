"""
The data collected here will be directly used as input to the simulator.
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils_data


# load data
dataset01, dataset0, dataset1 = utils_data.get_dataset()
datasets = [dataset01, dataset0, dataset1]

# transactions per hour of day
trans_per_hour0 = dataset0["Date"].apply(lambda date: date.hour).value_counts(sort=False)
trans_per_hour1 = dataset1["Date"].apply(lambda date: date.hour).value_counts(sort=False)
trans_per_hour = np.zeros((24, 2))
trans_per_hour[trans_per_hour0.index, 0] = trans_per_hour0
trans_per_hour[trans_per_hour1.index, 1] = trans_per_hour1
trans_per_hour /= np.sum(trans_per_hour, axis=0)
np.save('aggregated/trans_per_hour', trans_per_hour)

# transactions per month
trans_per_month0 = dataset0["Date"].apply(lambda date: date.month).value_counts(sort=False)
trans_per_month1 = dataset1["Date"].apply(lambda date: date.month).value_counts(sort=False)
trans_per_month = np.zeros((12, 2))
trans_per_month[trans_per_month0.index-1, 0] = trans_per_month0
trans_per_month[trans_per_month1.index-1, 1] = trans_per_month1
trans_per_month /= np.sum(trans_per_month, axis=0)
np.save('aggregated/trans_per_month', trans_per_month)



# data_stats_cols = ['ALL', 'NON-FRAUD', 'FRAUD']
#
# # --- CUSTOMER / FRAUDSTER STATISTICS ---
#
# # -> regarding countries
#
# # group countries by card
# country_per_card01 = dataset01.groupby(['CardID', 'Country']).size().reset_index()
# country_per_card0 = dataset0.groupby(['CardID', 'Country']).size().reset_index()
# country_per_card1 = dataset1.groupby(['CardID', 'Country']).size().reset_index()
#
# # get the probability of country per customer
# country_per_card_prob = utils_data.get_transaction_prob('Country', country_per_card01, country_per_card0, country_per_card1)
#
# # save
# country_per_card_prob.to_csv('./aggregated/country_per_customer_prob.csv', index_label=False)
#
# # print
# print("Country per customer prob: \n\n", country_per_card_prob.head(), "\n")
#
# # --> regarding currencies, given the country
#
# # group countries by card
# currency_per_country01 = dataset01.groupby(['Country', 'Currency']).size().reset_index()
# currency_per_country0 = dataset0.groupby(['Country', 'Currency']).size().reset_index()
# currency_per_country1 = dataset1.groupby(['Country', 'Currency']).size().reset_index()
#
# # get the probability of country per customer
# num_trans = pandas.DataFrame(0, index=currency_per_country01['Currency'].value_counts().index, columns=data_stats_cols)
# num_trans['ALL'] = currency_per_country01['Currency'].value_counts()
# num_trans['NON-FRAUD'] = d0[col_name].value_counts()
# num_trans['FRAUD'] = d1[col_name].value_counts()
# num_trans = num_trans.fillna(0)
# num_trans /= np.sum(num_trans, axis=0)
#
# # save
# country_prob.to_csv('./aggregated/country_currency_prob.csv', index_label=False)
#
# # print
# print("Currency per country prob: \n\n", country_prob.head(), "\n")
#
# # --- CURRENCY ---
#
# # how the merchant and the currency are tied together
# plt.figure()
# currencies = dataset01['Currency'].unique()
# merchants = dataset01['MerchantID'].unique()
# for curr_idx in range(len(currencies)):
#     for merch_idx in range(len(merchants)):
#         if currencies[curr_idx] in dataset01.loc[dataset01['MerchantID'] == merch_idx, 'Currency'].values:
#             plt.plot(curr_idx, merch_idx, 'ko')
#         plt.plot(range(len(currencies)), np.zeros(len(currencies))+merch_idx, 'r-', markersize=0.3)
# plt.xticks(range(len(currencies)), currencies)
# plt.savefig('./aggregated/merchant_vs_currency')
# plt.close()
#
# # --- AMOUNT ---
#
# # # plot transaction activity over time
# # plt.figure(figsize=(15, 12))
# # plt_idx = 1
# # print(dataset0["Country"].value_counts()[-5:])
# # print(dataset1["Country"].value_counts()[-5:])
# # for c in ['FR', 'NO', 'DE', 'DK', 'US', 'CA', 'GB']:
# #     for d in datasets:
# #         plt.subplot(7, 3, plt_idx)
# #         # dates = d["Date"].apply(lambda date: date.datetime())
# #         # dates = matplotlib.dates.date2num(dates)
# #         amounts = d.loc[d['Country'] == c, "Amount"]
# #         plt.plot(amounts.values, '.')
# #         plt_idx += 1
# #         plt.title(d.name)
# #         plt.xlabel('days (05.12.15 - 02.01.17)')
# #         plt.xticks([])
# #         if plt_idx == 2:
# #             plt.ylabel('num transactions')
# # plt.savefig('./aggregated/amount_per_transaction_over_time')
# # plt.close()
#
#
#
# # for col in ['Country']:
# #     for row in ['Currency']:
# #     plt.figure(figsize=(15, 10))
# #     d = dataset01
# #     currencies = d[col].unique()
# #     merchants = d[].unique()
# #     for c_idx in range(len(currencies)):
# #         for merch_idx in range(len(merchants)):
# #             # print(currencies[c_idx])
# #             # print(d.loc[d['MerchantID'] == merch_idx, 'Currency'])
# #             if currencies[c_idx] in d.loc[d['Currency'] == merchants[merch_idx], col].values:
# #                 plt.plot(c_idx, merch_idx, 'ko')
# #             plt.plot(range(len(currencies)), np.zeros(len(currencies))+merch_idx, 'r-', markersize=0.3)
# #     plt.xticks(range(len(currencies)), currencies)
# #     plt.savefig('./aggregated/currency_vs_{}'.format(col))
# #     plt.close()
#
# for col in ['CardID', 'Amount']:
#     plt.figure(figsize=(15, 5))
#     plt_idx = 1
#     for d in datasets:
#         plt.subplot(1, 3, plt_idx)
#         plt.plot(d[col], d["MerchantID"], '.')
#         # plt.xticks(d[col], d[col].unique())
#         plt_idx += 1
#     plt.savefig('./aggregated/merchant_vs_{}'.format(col))
#
# 1/0
#
# plt.figure()
# bins = [0, 5, 15, 30, 50, 5000, 10600]
# plt_idx = 1
# for d in datasets:
#     amount_counts, loc = np.histogram(d["Amount"], bins=bins)
#     amount_counts = np.array(amount_counts, dtype=np.float)
#     amount_counts /= np.sum(amount_counts)
#     plt.subplot(1, 3, plt_idx)
#     am_bot = 0
#     for i in range(len(amount_counts)):
#         plt.bar(plt_idx, amount_counts[i], bottom=am_bot, label='{}-{}'.format(bins[i], bins[i+1]))
#         am_bot += amount_counts[i]
#     plt_idx += 1
# plt.legend()
# plt.title("Amount distribution")
# plt_idx += 1
# plt.savefig('./aggregated/amount_distribution')
# plt.close()
#
# plt_idx = 1
# for i in range(len(bins)-1):
#     amounts = dataset0.loc[np.logical_and(bins[i] < dataset0["Amount"], dataset0["Amount"] < bins[i+1]), "Amount"]
#     plt.subplot(1, len(bins)-1, plt_idx)
#     plt.hist(amounts, bins=50)
#     plt_idx += 1
# plt.show()
#
# # --- COUNTRIES ---
#
# # # make some analysis only on france
# # dataset01_france = dataset01.loc[dataset01["Country"] == 'FR']
# # dataset0_france = dataset0.loc[dataset0["Country"] == 'FR']
# # dataset1_france = dataset1.loc[dataset1["Country"] == 'FR']
# # # plot transaction prob
# # trans_prob_france = get_transaction_prob("Amount", dataset01_france, dataset0_france, dataset1_france)
# # print(trans_prob_france.head())
# # # plot_bar_trans_prob(trans_prob_france, 'Currency', file_name='currency_FR')
# #
# # # make some analysis only on the US
# # dataset01_us = dataset01.loc[dataset01["Country"] == 'US']
# # dataset0_us = dataset0.loc[dataset0["Country"] == 'US']
# # dataset1_us = dataset1.loc[dataset1["Country"] == 'US']
# # # plot transaction prob
# # trans_prob_us = get_transaction_prob("Amount", dataset01_us, dataset0_us, dataset1_us)
# # # plot_bar_trans_prob(trans_prob_us, 'Currency', file_name='currency_US')
#
#
# # -- CURRENCIES ---
#
# # get the transactions per currency
# trans_prob_currency = get_transaction_prob("Currency")
# # save
# trans_prob_currency.to_csv('./aggregated/currency_trans_prob.csv', index_label=False)
# # plot
# plot_bar_trans_prob(trans_prob_currency, 'Currency')
# # print
# print("Currency prob: \n\n", trans_prob_currency.head(), "\n")
#
# # check how many currencies there are per country
# grouped_prob_country_currency = get_grouped_prob("Country", "Currency")
# print("Currency prob per country: \n\n", grouped_prob_country_currency.head(), '\n')
#
# # --- MERCHANTS ---
#
# merch_trans_frac = get_transaction_prob("MerchantID")
# plot_hist_num_transactions(merch_trans_frac, "MerchantID")
#
# # for the four busiest merchants, plot the activity over time
# num_busy_merchants = 4
# busy_merchants = merch_trans_frac.index[:num_busy_merchants]
# # plot transaction activity over time
# plt.figure(figsize=(13, 7))
# plt_idx = 1
# min_date_num = matplotlib.dates.date2num(min(dataset01["Date"]))
# max_date_num = matplotlib.dates.date2num(max(dataset01["Date"]))
# for m in busy_merchants:
#     for d in datasets:
#         plt.subplot(num_busy_merchants, 3, plt_idx)
#         dates = d.loc[d["MerchantID"] == m, "Date"].apply(lambda date: date.date())
#         all_trans = dates.value_counts(sort=False)
#         date_num = matplotlib.dates.date2num(all_trans.index)
#         plt.plot(date_num, all_trans.values, '.')
#         plt.xticks([])
#         plt.xlim([min_date_num, max_date_num])
#         if plt_idx < 4:
#             plt.title(d.name)
#         if plt_idx > (num_busy_merchants-1)*3:
#             plt.xlabel('days (05.12.15 - 02.01.17)')
#         if plt_idx % 3 == 1:
#             plt.ylabel('merchant {}'.format(m))
#         plt_idx += 1
# plt.tight_layout()
# plt.savefig('./aggregated/MerchantID_num-trans_time_top4')
# plt.close()
#
# # --- CREDIT CARDS ---
#
# card_counts = get_transaction_dist("CardID")
#
# # add to data_stats
# data_stats = pandas.concat([data_stats, card_counts])
#
# # plot
# plt.figure()
# bottoms = np.vstack((np.zeros(3), np.cumsum(card_counts, axis=0)))
# for i in range(card_counts.shape[0]):
#     plt.bar((0, 1, 2), card_counts.values[i], label=card_counts.index[i], bottom=bottoms[i])
# plt.xticks([0, 1, 2], ['all', 'non-fraud', 'fraud'])
# h = plt.ylabel('%')
# h.set_rotation(0)
# plt.title("Card Distribution")
# plt.legend()
# plt.savefig('./aggregated/card_distribution')
# plt.close()
#
# # # distribution of how many transactions are made by card
# # transactions_per_card = dataset01["CardID"].value_counts().value_counts()
# # print(transactions_per_card)
# # a, b,  _, _ = beta.fit(transactions_per_card)
# # plt.plot(transactions_per_card.index, transactions_per_card.values, '.')
# # plt.plot(beta.pdf(np.linspace(1, 200, 200), a, b), 'r--')
# # plt.xlabel('num transactions')
# # plt.ylabel('num credit cards')
# # plt.show()
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.plot(dataset01["CardID"], dataset01["MerchantID"], '.', markersize=1)
# plt.xlabel('card')
# plt.ylabel('merchant')
# plt.subplot(1, 3, 2)
# plt.plot(dataset0["CardID"], dataset0["MerchantID"], '.', markersize=1)
# plt.xlabel('card')
# plt.subplot(1, 3, 3)
# plt.plot(dataset1["CardID"], dataset1["MerchantID"], '.', markersize=1)
# plt.xlabel('card')
# plt.savefig('./aggregated/card_merchant_relation')
#
# # --- DATE ---
#
#
#
# # ----------------------
# # --- aggregate data ---
# # ----------------------
#
#
#
# print("")
#
# for d in datasets:
#
#     agent_ids = d['MerchantID'].unique()
#     agent_stats = pandas.DataFrame(columns=range(len(agent_ids)))
#
#     agent_stats.loc['genuine transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 0) for c in agent_ids]
#     agent_stats.loc['fraudulent transactions'] = [sum(d.loc[d["MerchantID"] == c, "Target"] == 1) for c in agent_ids]
#     agent_stats.loc['credit cards'] = [len(d.loc[d["MerchantID"] == c, "CardID"].unique()) for c in agent_ids]
#     agent_stats.loc['countries'] = [len(d.loc[d["MerchantID"] == c, "Country"].unique()) for c in agent_ids]
#     agent_stats.loc['min amount'] = [min(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]
#     agent_stats.loc['max amount'] = [max(d.loc[d["MerchantID"] == c, "Amount"]) for c in agent_ids]
#
#     print(agent_stats)
#     print("")
#
#     agent_stats.to_csv('./merchStats_{}.csv'.format(d.name), index_label=False)
