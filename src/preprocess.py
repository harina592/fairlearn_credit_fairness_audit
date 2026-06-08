from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    le = LabelEncoder()

    for col in df.columns:

        # Encode both object and category types
        if df[col].dtype == "object" or str(df[col].dtype) == "category":

            df[col] = le.fit_transform(df[col].astype(str))

    return df