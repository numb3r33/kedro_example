import pandas as pd

def preprocess_train(train: pd.DataFrame) -> pd.DataFrame:
    obj_cols = train.select_dtypes(include='object').columns

    for col in obj_cols:
        train.loc[:, col] = pd.factorize(train.loc[:, col], sort=True)[0]

    train = train.drop('Unnamed: 0', axis=1)

    return train

def preprocess_test(test: pd.DataFrame) -> pd.DataFrame:
    obj_cols = test.select_dtypes(include='object').columns

    for col in obj_cols:
        test.loc[:, col] = pd.factorize(test.loc[:, col], sort=True)[0]

    test = test.drop('Unnamed: 0', axis=1)

    return test









