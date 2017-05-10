import pandas
from currency_converter import CurrencyConverter
from datetime import datetime
from pytz import timezone, country_timezones
from sklearn import preprocessing

# read in dataset
dataset = pandas.read_csv('./anonymized_dataset.csv')
print(dataset.head())

# throw away the columns I won't use
del dataset["Unnamed: 0"]   # this is just an arbitrary index
del dataset["id"]           # because I don't know what this is
del dataset["AccountID"]    # because I don't know what this is
del dataset["Email"]        # not relevant

# merge first and last name
dataset["Name"] = dataset["First Name"].map(str) + dataset["Last Name"]
del dataset["First Name"]
del dataset["Last Name"]

dataset = dataset.rename(columns={'Merchant': 'MerchantID', 'Card': 'CardID', 'date': 'Date', 'target': 'Target',
                                  'GeoCode': 'PurchaseCountry'})

print(dataset.columns)

# convert Merchant, Card, FirstName, LastName, Email into integers
le = preprocessing.LabelEncoder()
dataset["MerchantID"] = le.fit_transform(dataset["MerchantID"])
dataset["CardID"] = le.fit_transform(dataset["CardID"])
dataset["Name"] = le.fit_transform(dataset["Name"])

# where the GeoCode is unknown, use the card issuing Country (important for time conversion)
dataset.loc[dataset["PurchaseCountry"] == "--", "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"] == "--", "Country"]
dataset.loc[dataset["PurchaseCountry"].isnull(), "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"].isnull(), "Country"]

# convert dates into datetime format
dataset["Date"] = dataset["Date"].apply(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))

# add timezone info (Pacific Standard Time)
dataset["Date"] = dataset["Date"].apply(lambda date: date.replace(tzinfo=timezone('US/Pacific')))

# convert dates into local times (not that if it's ambiguous like Australia I just take the first entry
dataset["Date"] = dataset.apply(lambda d: d["Date"].astimezone(timezone(country_timezones(d["PurchaseCountry"])[0])), axis=1)

# convert currencies into EUR (using the conversion rate from the date of purchase)
c = CurrencyConverter(fallback_on_missing_rate=True)
dataset["Amount"] = dataset.apply(lambda d: round(c.convert(d["Amount"], d["Currency"], 'EUR', d["Date"]), 2), axis=1)
# del dataset["Currency"]

print(dataset.head())

dataset.to_csv('./anonymized_dataset_preprocessed.csv', index_label=False)
