import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import joblib

if __name__ == "__main__":

    # seed global
    seed = 42

    # Carrega os dados
    df = pd.read_csv('base/loan_default.csv')

    # Identifica no dataset as variáveis independentes e a variavel alvo
    independentcols = ['loan_amount', 'property_value', 'income', 'LTV', 'loan_limit', 'Credit_Worthiness',
                              'Neg_ammortization', 'Gender', 'loan_type', 'credit_type', 'Region', 'approv_in_adv',
                              'business_or_commercial', 'lump_sum_payment', 'co_applicant_credit_type',
                              'submission_of_application']
    targetcol_inad = 'Status'
    targetcol_cluster = 'Cluster'

    # Trata os dados
    df_ajustado = df[['loan_amount', 'property_value', 'income', 'LTV', 'loan_limit', 'Credit_Worthiness',
                              'Neg_ammortization', 'Gender', 'loan_type', 'credit_type', 'Region', 'approv_in_adv',
                              'business_or_commercial', 'lump_sum_payment', 'co_applicant_credit_type',
                              'submission_of_application', 'Status']]

    df_ajustado.dropna(subset=['loan_limit', 'approv_in_adv', 'Neg_ammortization', 'submission_of_application'], axis=0, inplace=True)

    colunas = ['income', 'property_value', 'LTV']
    for coluna in colunas:
        df_ajustado[coluna].fillna(df_ajustado[coluna].mean(), inplace=True)

    df_ajustado.reset_index(drop=True, inplace=True)
    df_ajustado['Gender'].replace(['Sex Not Available'], df_ajustado['Gender'].mode(), inplace=True)

    # modelo01 - Clusterização
    df_cluster = df_ajustado[independentcols]

    #scaling
    df_numerico = df_cluster.select_dtypes(include=['int64', 'float64'])
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numerico), columns=df_numerico.columns)
    df_categorico = df_cluster.select_dtypes(include=[object])
    df_cluster = df_scaled.join(df_categorico)

    # binary encoding
    colunas = [i for i in df_cluster.columns if df_cluster[i].dtype == 'object' and df_cluster[i].nunique() == 2]
    for i in colunas:
        df_cluster[i].replace([i for i in df_cluster[i].unique()], [0, 1], inplace=True)

    # get_dummies
    df_cluster = pd.get_dummies(df_cluster)

    # clusterização
    kmeans_Standard_k2 = KMeans(n_clusters=2, random_state=seed, n_init='auto')
    labels_Standard_k2 = kmeans_Standard_k2.fit_predict(df_cluster)
    df_cluster[targetcol_cluster] = labels_Standard_k2

    # separação da base em treino e teste
    X = df_cluster.drop(targetcol_cluster, axis=1)
    y = df_cluster[targetcol_cluster]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    # criação do modelo
    moodelo_cluster = RandomForestClassifier(max_depth=30, n_estimators=200, n_jobs=-1, random_state=seed)
    moodelo_cluster.fit(X=X_train, y=y_train)
    moodelo_cluster.independentcols = independentcols
    moodelo_cluster_acuracia = moodelo_cluster.score(X=X_test, y=y_test)
    print("Modelo 01 (cluster), criado com acurácia de: [{0}]".format(moodelo_cluster_acuracia))

    # modelo02 - Classificação Inadimplência
    df_inad = df_ajustado[independentcols]

    # scaling
    df_numerico = df_inad.select_dtypes(include=['int64', 'float64'])
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numerico), columns=df_numerico.columns)
    df_categorico = df_inad.select_dtypes(include=[object])
    df_inad = df_scaled.join(df_categorico)
    df_inad[targetcol_inad] = df_ajustado[targetcol_inad]

    # binary encoding
    colunas = [i for i in df_inad.columns if df_inad[i].dtype == 'object' and df_inad[i].nunique() == 2]
    for i in colunas:
        df_inad[i].replace([i for i in df_inad[i].unique()], [0, 1], inplace=True)

    # get_dummies
    df_inad = pd.get_dummies(df_inad)

    # some
    smote = SMOTE(random_state=seed)
    X_smote, y_smote = smote.fit_resample(df_inad.drop(targetcol_inad, axis=1), df_inad[targetcol_inad])
    df_inad = pd.concat([X_smote, y_smote], axis=1)

    # separação da base em treino e teste
    X = df_inad.drop(targetcol_inad, axis=1)
    y = df_inad[targetcol_inad]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    # criação do modelo
    moodelo_inad = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    moodelo_inad.fit(X=X_train, y=y_train)
    moodelo_inad.independentcols = independentcols
    moodelo_inad_acuracia = moodelo_inad.score(X=X_test, y=y_test)
    print("Modelo 02 (inadimplência), criado com acurácia de: [{0}]".format(moodelo_inad_acuracia))

    # Salva ambos os modelos
    joblib.dump(moodelo_cluster, 'models/modelo01.joblib')
    print("Modelo 01 (classificador) salvo com sucesso.")
    joblib.dump(moodelo_inad, 'models/modelo02.joblib')
    print("Modelo 02 (classificador) salvo com sucesso.")
    pass
