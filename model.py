from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle
import pandas as pd

df = pd.read_csv('tucarro_dataset_final2.csv')

df.loc[df.comb_type.str.lower().str.contains("habrido"), "comb_type"] = "hibrido"
df.loc[df.comb_type.str.lower().str.contains("diasel"), "comb_type"] = "diesel"

catcodes = {}

for i in ['brand', 'model', 'color', 'comb_type', 'trans', 'body']:
    df[i] = df[i].astype("category")
    catcodes[i] = dict(zip(df[i], df[i].cat.codes))
    df[i] = df[i].cat.codes

with open('catodes_train.pickle', 'wb') as handle:
    pickle.dump(catcodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

df['price'] = df['price'].astype("float64")


#x is the model that doesn't have the price attribute
# y is the one that only contains the price
x = df.drop(['price', 'doors'], axis=1).values
y = df['price'].values

# Dataset has been divided into several datatest into diferent proportion validation is 15%, testing is 15%, and the training is 70%
validation_ratio = 0.15
test_ratio = 0.15
train_ratio = 1 - validation_ratio + test_ratio


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=43)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_ratio, random_state=43)


pandas_train = pd.DataFrame(x_train)

modern_cars_x_train = pandas_train[pandas_train[2] >= 2000]
modern_cars_x_train = modern_cars_x_train.to_numpy()
modern_cars_y_train = pd.DataFrame(y_train).iloc[pandas_train[pandas_train[2] >= 2000].index, :]

old_cars_x_train = pandas_train[pandas_train[2] < 2000]
old_cars_x_train = old_cars_x_train.to_numpy()
old_cars_y_train = pd.DataFrame(y_train).iloc[pandas_train[pandas_train[2] < 2000].index, :]


kitty_modern = CatBoostRegressor(num_trees=1790, learning_rate=0.203, depth=6)
kitty_modern.fit(modern_cars_x_train, modern_cars_y_train)

kitty_old = CatBoostRegressor(num_trees=1790, learning_rate=0.203, depth=6)
kitty_old.fit(modern_cars_x_train, modern_cars_y_train)

with open('kitty_modern.pkl', 'wb') as cat_modern:
    pickle.dump(kitty_modern, cat_modern)

with open('kitty_old.pkl', 'wb') as cat_old:
    pickle.dump(kitty_old, cat_old)

with open('modern_data.pkl', 'wb') as modern_data:
    pickle.dump(modern_cars_x_train, modern_data)

with open('old_data.pkl', 'wb') as old_data:
    pickle.dump(old_cars_x_train, old_data)
