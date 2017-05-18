"""
Let this run if you have the original dataset

    anonymized_dataset.csv
    
to produce the preprocessed log file of transactions

    transaction_log.csv
    
"""
from datetime import datetime
import numpy as np
import pandas
from currency_converter import CurrencyConverter
from pytz import timezone, country_timezones
from sklearn import preprocessing

dataset = pandas.read_csv('anonymized_dataset.csv')
print(dataset.head())
print("")

# throw away the columns we don't use
del dataset["Unnamed: 0"]   # this is just an arbitrary index
del dataset["id"]           # because I don't know what this is
del dataset["AccountID"]    # because I don't know what this is
del dataset["Email"]        # not relevant

# merge first and last name
dataset["Name"] = dataset["First Name"].map(str) + dataset["Last Name"]
del dataset["First Name"]
del dataset["Last Name"]

# rename the column names for consistency
dataset = dataset.rename(columns={'Merchant': 'MerchantID', 'Card': 'CardID', 'date': 'Date', 'target': 'Target',
                                  'GeoCode': 'PurchaseCountry'})

# convert Merchant, Card, FirstName, LastName, Email into integers
le = preprocessing.LabelEncoder()
dataset["MerchantID"] = le.fit_transform(dataset["MerchantID"])
dataset["CardID"] = le.fit_transform(dataset["CardID"])
dataset["Name"] = le.fit_transform(dataset["Name"])

# convert dates into datetime format
dataset["Date"] = dataset["Date"].apply(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))

# where the GeoCode is unknown, use the card issuing Country (important for time conversion)
dataset.loc[dataset["PurchaseCountry"] == "--", "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"] == "--", "Country"]
dataset.loc[dataset["PurchaseCountry"].isnull(), "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"].isnull(), "Country"]

# add timezone info (Pacific Standard Time)
dataset["Date"] = dataset["Date"].apply(lambda date: date.replace(tzinfo=timezone('US/Pacific')))

# convert dates into local times (not that if it's ambiguous like Australia I just take the first entry
dataset["Date"] = dataset.apply(lambda d: d["Date"].astimezone(timezone(country_timezones(d["PurchaseCountry"])[0])), axis=1)

# remove the timezone info
dataset["Date"] = dataset.apply(lambda d: d["Date"].replace(tzinfo=None), axis=1)

# convert currencies into EUR (using the conversion rate from the date of purchase)
c = CurrencyConverter(fallback_on_missing_rate=True)
dataset["Amount"] = dataset.apply(lambda d: round(c.convert(d["Amount"], d["Currency"], 'EUR', d["Date"]), 2), axis=1)

# the data goes from 12.5.15 to 2.1.17
# only three fraud cases are recorded in 2016, so we reduce to 01.01.16 - 31.12.16
dataset = dataset[dataset['Date'].apply(lambda d: d.year) == 2016]

# compare number of unique credit cards to number of unique names
print("num cards (all): ", dataset["CardID"].unique().shape[0])
print("num names (all): ", dataset["Name"].unique().shape[0])
print("num cards (genuine): ", dataset.loc[dataset["Target"] == 0, "CardID"].unique().shape[0])
print("num names (genuine): ", dataset.loc[dataset["Target"] == 0, "Name"].unique().shape[0])
print("num cards (fraud): ", dataset.loc[dataset["Target"] == 1, "CardID"].unique().shape[0])
print("num names (fraud): ", dataset.loc[dataset["Target"] == 1, "Name"].unique().shape[0])
print("")
# --> high overlap, we remove the names

# since the overlap between unique cards and unique names is very high, we remove the names
del dataset["Name"]

# analyse how often the card issuing country and purchase country differ
print("(card != purchase country) - (all): ", 100*round(np.sum(dataset["Country"] != dataset["PurchaseCountry"])/dataset.shape[0], 4), '%')
print("(card != purchase country) - (genuine): ", 100*round(np.sum(dataset.loc[dataset["Target"] == 0, "Country"] != dataset.loc[dataset["Target"] == 0, "PurchaseCountry"])/sum(dataset["Target"] == 0), 4), '%')
print("(card != purchase country) - (fraud): ", 100*round(np.sum(dataset.loc[dataset["Target"] == 1, "Country"] != dataset.loc[dataset["Target"] == 1, "PurchaseCountry"])/sum(dataset["Target"] == 1), 4), '%')
print("")
# --> we will delete PurchaseCountry after converting time zones!

# we only keep the 'card country' as a feature because the card and purchase
# country are mostly overlapping, and the purchase country data is incomplete
del dataset["PurchaseCountry"]

# bring columns into convenient order
dataset = dataset[['Date', 'CardID', 'MerchantID', 'Amount', 'Currency', 'Country', 'Target']]

print(dataset.head())

dataset.to_csv('transaction_log.csv', index_label=False)
