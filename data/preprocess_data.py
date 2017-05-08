import pandas
import numpy as np
from currency_converter import CurrencyConverter
from datetime import datetime
from pytz import timezone, country_timezones
from sklearn import preprocessing

# read in dataset
dataset = pandas.read_csv('./anonymized_dataset.csv')
# dataset = dataset[:100000]
print(dataset.head())

# throw away the first column and the transaction ID
del dataset["Unnamed: 0"]
del dataset["id"]

# convert AccountID, Merchant, Card, FirstName, LastName, Email into integers
le = preprocessing.LabelEncoder()
dataset["AccountID"] = le.fit_transform(dataset["AccountID"])
dataset["Merchant"] = le.fit_transform(dataset["Merchant"])
dataset["Card"] = le.fit_transform(dataset["Card"])
dataset["First Name"] = le.fit_transform(dataset["First Name"])
dataset["Last Name"] = le.fit_transform(dataset["Last Name"])
dataset["Email"] = le.fit_transform(dataset["Email"])

# where the GeoCode is unknown, just use the Country
dataset.loc[dataset["GeoCode"] == "--", "GeoCode"] = dataset.loc[dataset["GeoCode"] == "--", "Country"]
dataset.loc[dataset["GeoCode"].isnull(), "GeoCode"] = dataset.loc[dataset["GeoCode"].isnull(), "Country"]

# convert dates into datetime format
dataset["date"] = dataset["date"].apply(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))

# add timezone info (Pacific Standard Time)
dataset["date"] = dataset["date"].apply(lambda date: date.replace(tzinfo=timezone('US/Pacific')))

# convert dates into local times (not that if it's ambiguous like Australia I just take the first entry
dataset["date"] = dataset.apply(lambda d: d["date"].astimezone(timezone(country_timezones(d["GeoCode"])[0])), axis=1)

# convert currencies into EUR
c = CurrencyConverter(fallback_on_missing_rate=True)
dataset["Amount"] = dataset.apply(lambda d: c.convert(d["Amount"], 'EUR', d["Currency"], d["date"]), axis=1)
# del dataset["Currency"]

print(dataset.head())

dataset.to_csv('./anonymized_dataset_preprocessed.csv')
