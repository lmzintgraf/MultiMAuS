import pandas
from currency_converter import CurrencyConverter
from datetime import datetime
from pytz import timezone, country_timezones
from sklearn import preprocessing

# read in dataset
dataset = pandas.read_csv('./anonymized_dataset.csv')
print(dataset.head())

# throw away the first column and the transaction ID
del dataset["Unnamed: 0"]
del dataset["id"]

dataset = dataset.rename(columns={'Merchant': 'MerchantID', 'Card': 'CardID', 'date': 'Date',
                        'target': 'Target', 'First Name': 'FirstName', 'Last Name': 'LastName',
                        'Country': 'CardCountry', 'GeoCode': 'PurchaseCountry'})

print(dataset.columns)

# convert AccountID, Merchant, Card, FirstName, LastName, Email into integers
le = preprocessing.LabelEncoder()
dataset["AccountID"] = le.fit_transform(dataset["AccountID"])
dataset["MerchantID"] = le.fit_transform(dataset["MerchantID"])
dataset["CardID"] = le.fit_transform(dataset["CardID"])
dataset["FirstName"] = le.fit_transform(dataset["FirstName"])
dataset["LastName"] = le.fit_transform(dataset["LastName"])
dataset["Email"] = le.fit_transform(dataset["Email"])

# where the GeoCode is unknown, use the Country
dataset.loc[dataset["PurchaseCountry"] == "--", "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"] == "--", "CardCountry"]
dataset.loc[dataset["PurchaseCountry"].isnull(), "PurchaseCountry"] = dataset.loc[dataset["PurchaseCountry"].isnull(), "CardCountry"]

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
