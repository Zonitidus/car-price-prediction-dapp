import streamlit as st
import pickle
import pandas as pd

df = pd.read_csv('tucarro_dataset_final2.csv')

df.loc[df.comb_type.str.lower().str.contains("habrido"), "comb_type"] = "hibrido"
df.loc[df.comb_type.str.lower().str.contains("diasel"), "comb_type"] = "diesel"

df['brand'] = df['brand'].astype("category")
df['model'] = df['model'].astype("category")
df['color'] = df['color'].astype("category")
df['comb_type'] = df['comb_type'].astype("category")
df['trans'] = df['trans'].astype("category")
df['body'] = df['body'].astype("category")
df['price'] = df['price'].astype("float64")

with open('kitty_modern.pkl', 'rb') as cat_modern:
    kitty_modern = pickle.load(cat_modern)

with open('kitty_old.pkl', 'rb') as cat_old:
    kitty_old = pickle.load(cat_old)

with open('catodes_train.pickle', 'rb') as handle:
    catcodes_train = pickle.load(handle)


def predict_modern(df_feats):
    df_feats = input_formatter(df_feats)
    return output_formatter(kitty_modern.predict(df_feats))


def predict_old(df_feats):
    df_feats = input_formatter(df_feats)
    return output_formatter(kitty_old.predict(df_feats))


def input_formatter(df_feats):
    df_feats.columns = df_feats.columns.str.lower()
    df_feats.rename(columns={'combustion type': 'comb_type', 'transmission': 'trans'}, inplace=True)

    catcoded_df = df_feats.copy()

    for i in ['brand', 'model', 'color', 'comb_type', 'trans', 'body']:
        catcoded_df[i].values[0] = catcodes_train[i][df_feats[i].values[0]]

    return catcoded_df

def output_formatter(pred):
    return "$ {:,.0f}".format(int(pred[0]))


def main():
    st.title("¿Cuánto cuesta mi auto?")
    st.sidebar.header("Parámetros")

    brand_df = df['brand'].drop_duplicates()

    color_df = df['color'].drop_duplicates()
    combustion_df = df['comb_type'].drop_duplicates()
    transmission_df = df['trans'].drop_duplicates()
    body_df = df['body'].drop_duplicates()

    def init_parameters():
        brands = st.sidebar.selectbox('Brand', brand_df)

        model_df = df['model'].drop_duplicates()

        models = st.sidebar.selectbox('Model', model_df)

        years = st.sidebar.slider('Year', 1969, 2022, 2017, 1)
        colors = st.sidebar.selectbox('Color', color_df)
        combustions = st.sidebar.selectbox('Combustion type', combustion_df)
        transmissions = st.sidebar.selectbox('Transmission type', transmission_df)
        motors = st.sidebar.slider('Motor capacity', 0.5, 10.0, 1.8, 0.1)
        bodies = st.sidebar.selectbox('Body', body_df)
        kms = st.sidebar.slider('Km', 0, 500000, 0, 100)

        data = {'Brand': brands,
                'Model': models,
                'Year': years,
                'Color': colors,
                'Combustion type': combustions,
                'Transmission': transmissions,
                'Motor': motors,
                'Body': bodies,
                'Km': kms
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df_st = init_parameters()
    st.write(df_st)

    if st.button('Predict'):

        if df_st['Year'].to_numpy()[0] > 1999:
            st.success(predict_modern(df_st))
        else:
            st.success(predict_old(df_st))


if __name__ == '__main__':
    main()
