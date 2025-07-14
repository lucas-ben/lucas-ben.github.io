import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

df = pd.read_csv("/Users/lucasben/Desktop/Data-ATP-W119-PEW.csv", low_memory=False)
# mapping dictionary for variable replacement
var_labels = {
    'AIHCCOMF_W119': 'Comfort with AI in healthcare',
    'HCMEDBIAS_W119': 'Concern about bias in healthcare',
    'AIHCCHG_QUAL_W119': 'AI impact on quality of care',
    'AIHCCHG_MIST_W119': 'AI impact on medical mistakes',
    'AIHCCHG_REL_W119': 'AI impact on doctor-patient relationship',
    'AIHCCHG_RACETHN_W119': 'AI impact on fairness in race/ethnicity',
    'AIHCCHG_SECUR_W119': 'AI impact on health data security',
    'AIKNOW_INDEX_W119': 'AI knowledge score',
    'F_AGECAT': 'Age group',
    'F_RACECMB': 'Race',
    'F_EDUCCAT': 'Education level',
    'F_INC_SDT1': 'Household income',
    'F_IDEO': 'Political ideology',
    'F_GENDER': 'Gender',
    'F_RELIG': 'Religion',
    'F_REG': 'Geographic region',
    'F_INTFREQ': 'Internet usage frequency',
    'F_METRO': 'Metropolitan area',
    'F_CITIZEN': 'Citizenship status',
}

data = df[list(var_labels.keys())].rename(columns=var_labels)
data.isna().sum()
data.dtypes

# checking if variables are ordinal
for col in data.columns:
    print(f"{col}: {data[col].unique()}")

data.replace(['99', 99, ' '], np.nan, inplace=True)

df1 = data[[
    'Comfort with AI in healthcare',
    'Concern about bias in healthcare',
    'AI impact on quality of care',
    'AI impact on medical mistakes',
    'AI impact on doctor-patient relationship',
    'AI impact on fairness in race/ethnicity',
    'AI impact on health data security'
]]

df2 = data[['AI knowledge score']]

df3 = data[[
    'Age group',
    'Race',
    'Education level',
    'Household income',
    'Political ideology',
    'Gender',
    'Religion',
    'Geographic region',
    'Internet usage frequency',
    'Metropolitan area',
    'Citizenship status'
]]

# Building a ML model to use demographic and attitudinal data to predict a respondent's trust profile toward AI in healthcare
## ie: Identifying future behaviour based on observed patterns

## the outcome is a predicted classification
## the inputs are survey responses, either ordinal or categorical predictors 
## Using factor analysis to explore latent structures in the data which will help build the trust profiles which will later be used to classify new data with ML model


corr_matrix = df1.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
plt.title('Correlation Heatmap of AI Perceptions in Healthcare', fontsize=14)
plt.show()


from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2, max_iter=3000, random_state=42)
# dropping missing values for factor analysis
df1_clean = df1.dropna(subset=df1.columns)
# training the factor analysis model
fa.fit(df1_clean)
# projecting each observation onto the factor space
fa_scores = fa.transform(df1_clean)
# creating a dataframe with factor scores
factors = pd.DataFrame(
    fa_scores,
    columns=['F1_PositiveExpectancy', 'F2_RelationalImpact'],
    index=df1_clean.index
)


data_fa = data.loc[df1_clean.index].join(factors)

from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
fa = FactorAnalyzer(n_factors=df1.shape[1], rotation=None)
fa.fit(df1)

ev, v = fa.get_eigenvalues()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev)+1), ev, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')  
plt.grid(True)
plt.show()

print(pd.Series(ev, name="Eigenvalue"))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
data_fa['trust_profile'] = kmeans.fit_predict(data_fa[['F1_PositiveExpectancy',
                                                       'F2_RelationalImpact']])
centroids = kmeans.cluster_centers_
print(pd.DataFrame(
    centroids,
    columns=['F1_PositiveExpectancy', 'F2_RelationalImpact']
))


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=data_fa,
    x='F1_PositiveExpectancy',
    y='F2_RelationalImpact',
    hue='trust_profile',
    palette='Set1',
    alpha=0.6,
    s=50
)


centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    c='black', 
    s=200, 
    marker='X', 
    label='Centroids'
)


plt.title('KMeans Clustering on Factor Scores')
plt.xlabel('F1: Positive Expectancy')
plt.ylabel('F2: Relational Impact')
plt.legend(title='Trust Profile', loc='best')
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

kmeans3 = KMeans(n_clusters=2, random_state=42)
features3 = ['F1_PositiveExpectancy', 'F2_RelationalImpact', 'F3_SomeDimension']
data_fa['trust_profile3'] = kmeans3.fit_predict(data_fa[features3])

centroids3 = pd.DataFrame(
    kmeans3.cluster_centers_,
    columns=features3
)
print(centroids3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    data_fa['F1_PositiveExpectancy'],
    data_fa['F2_RelationalImpact'],
    data_fa['F3_SomeDimension'],
    c=data_fa['trust_profile3'],
    cmap='Set1',
    alpha=0.6,
    s=40
)

ax.scatter(
    centroids3['F1_PositiveExpectancy'],
    centroids3['F2_RelationalImpact'],
    centroids3['F3_SomeDimension'],
    c='black',
    s=200,
    marker='X',
    label='Centroids'
)

ax.set_xlabel('F1: Positive Expectancy')
ax.set_ylabel('F2: Relational Impact')
ax.set_zlabel('F3: Some Dimension')
ax.set_title('3D KMeans on Three Factors')
ax.legend()
plt.tight_layout()
plt.show()

for col in ['Geographic region','Internet usage frequency']:
    data_fa[col] = data_fa[col].astype(str)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

feature_cols = [
    'F1_PositiveExpectancy', 'F2_RelationalImpact',  
    'AI knowledge score',                           
    'Age group', 'Race', 'Education level', 'Household income',
    'Political ideology', 'Gender', 'Religion', 'Geographic region',
    'Internet usage frequency', 'Metropolitan area', 'Citizenship status'
]

X = data_fa.loc[df1_clean.index, feature_cols]
y = data_fa.loc[df1_clean.index, 'trust_profile']

numeric_feats = ['F1_PositiveExpectancy', 'F2_RelationalImpact', 'AI knowledge score']
ordinal_feats = ['Age group', 'Education level', 'Household income', 'Political ideology']
nominal_feats = [
    'Race', 'Gender', 'Religion', 'Geographic region',
    'Internet usage frequency', 'Metropolitan area', 'Citizenship status'
]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('to_str', FunctionTransformer(lambda X: X.astype(str))),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('ord', ordinal_transformer, ordinal_feats),
    ('nom', nominal_transformer, nominal_feats)
])

rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])

mlp_pipeline = Pipeline([
    ('prep', preprocessor),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation='relu',
                          solver='adam',
                          max_iter=300,
                          random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

rf_pipeline.fit(X_train, y_train)
rf_acc = rf_pipeline.score(X_test, y_test)
print(f"Random Forest test accuracy: {rf_acc:.3f}")

mlp_pipeline.fit(X_train, y_train)
mlp_acc = mlp_pipeline.score(X_test, y_test)
print(f"Neural Network test accuracy: {mlp_acc:.3f}")

import matplotlib.pyplot as plt
import seaborn as sns

numeric_feats = ['F1_PositiveExpectancy', 'F2_RelationalImpact', 'AI knowledge score']
params = X_train[numeric_feats].agg(['mean','std']).T
print(params)

for feat in numeric_feats:
    mu, sigma = params.loc[feat, 'mean'], params.loc[feat, 'std']
    sns.histplot(X_train[feat], stat='density', bins=30, color='skyblue', label='empirical')
    x = np.linspace(mu-4*sigma, mu+4*sigma, 200)
    plt.plot(x, 
             1/(sigma*np.sqrt(2*np.pi))*np.exp(- (x-mu)**2/(2*sigma**2)), 
             color='darkorange', label='normal fit')
    plt.title(f"{feat}\nμ={mu:.2f}, σ={sigma:.2f}")
    plt.legend()
    plt.show()

import numpy as np
import pandas as pd

n_sim = 1000

sim_num = {
    feat: np.random.normal(
        loc=params.loc[feat,'mean'],
        scale=params.loc[feat,'std'],
        size=n_sim
    )
    for feat in numeric_feats
}

def empirical_sampler(series: pd.Series, size: int):
    probs = series.value_counts(normalize=True)
    return np.random.choice(probs.index, size=size, p=probs.values)

ord_feats = ['Age group','Education level','Household income','Political ideology']
nom_feats = [
    'Race','Gender','Religion','Geographic region',
    'Internet usage frequency','Metropolitan area','Citizenship status'
]

sim_ord = {feat: empirical_sampler(X_train[feat], n_sim) for feat in ord_feats}
sim_nom = {feat: empirical_sampler(X_train[feat].astype(str), n_sim) for feat in nom_feats}

sim_df = pd.DataFrame({**sim_num, **sim_ord, **sim_nom})
print(sim_df.head())

sim_rf_preds = rf_pipeline.predict(sim_df)

sim_mlp_preds = mlp_pipeline.predict(sim_df)

print("RF preds:", sim_rf_preds[:10])
print("MLP preds:", sim_mlp_preds[:10])

import pandas as pd

rf_counts  = pd.Series(sim_rf_preds, name='RF Profile').value_counts().sort_index()
mlp_counts = pd.Series(sim_mlp_preds, name='MLP Profile').value_counts().sort_index()

print(rf_counts)
print(mlp_counts)

rf_probs = rf_pipeline.predict_proba(sim_df)

mlp_probs = mlp_pipeline.predict_proba(sim_df)

rf_conf  = rf_probs.max(axis=1).mean()
mlp_conf = mlp_probs.max(axis=1).mean()
print(f"RF avg confidence: {rf_conf:.2f}")
print(f"MLP avg confidence: {mlp_conf:.2f}")

import numpy as np
import pandas as pd

numeric_feats = ['F1_PositiveExpectancy', 'F2_RelationalImpact', 'AI knowledge score']
params = X_train[numeric_feats].agg(['mean','std']).T

def simulate_survey(params, n_sim=1000, jitter=0.05):
    """
    Returns a DataFrame of simulated survey data:
      - numeric_feats drawn from N(μ*(1+δ), σ*(1+δ))
      - ordinal & nominal sampled from X_train empirical distribs
    jitter: max relative noise on μ and σ (e.g. 5%)
    """
    sim_data = {}

    for feat, row in params.iterrows():
        mu, sigma = row['mean'], row['std']
        δ = np.random.normal(0, jitter)             # small relative shift
        sim_mu, sim_sigma = mu*(1+δ), sigma*(1+δ)
        sim_data[feat] = np.random.normal(sim_mu, sim_sigma, size=n_sim)

    def emp(col): 
        p = X_train[col].value_counts(normalize=True)
        return np.random.choice(p.index, size=n_sim, p=p.values)

    for f in ['Age group','Education level','Household income','Political ideology']:
        sim_data[f] = emp(f)

    for f in ['Race','Gender','Religion','Geographic region',
              'Internet usage frequency','Metropolitan area','Citizenship status']:
        sim_data[f] = emp(f).astype(str)

    return pd.DataFrame(sim_data)

from sklearn.metrics import accuracy_score

n_runs = 1000
rf_scores, mlp_scores = [], []

for i in range(n_runs):
    sim_df = simulate_survey(params, n_sim=1000, jitter=0.05)

    true_labels = kmeans.predict(sim_df[['F1_PositiveExpectancy','F2_RelationalImpact']])
    

    rf_pred  = rf_pipeline.predict(sim_df)
    mlp_pred = mlp_pipeline.predict(sim_df)
    

    rf_scores.append(accuracy_score(true_labels, rf_pred))
    mlp_scores.append(accuracy_score(true_labels, mlp_pred))

import matplotlib.pyplot as plt


rf_scores = pd.Series(rf_scores, name='RF Acc')
mlp_scores = pd.Series(mlp_scores, name='MLP Acc')

print("RF   mean±sd:", rf_scores.mean(), "±", rf_scores.std())
print("MLP  mean±sd:", mlp_scores.mean(), "±", mlp_scores.std())

print("RF   95% CI:", rf_scores.quantile([0.025,0.975]).values)
print("MLP  95% CI:", mlp_scores.quantile([0.025,0.975]).values)


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
rf_scores.hist(bins=20, color='skyblue')
plt.title("RF Accuracy Distribution")

plt.subplot(1,2,2)
mlp_scores.hist(bins=20, color='lightgreen')
plt.title("MLP Accuracy Distribution")

plt.tight_layout()
plt.show()


from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from sklearn.pipeline import Pipeline


def make_deep_model(input_dim,
                    layers=(256,128,64),
                    activation='relu',
                    dropout_rate=0.3,
                    lr=1e-3):

    inp = keras.Input(shape=(input_dim,))
    x = inp


    for units in layers:
        x = keras.layers.Dense(units, activation=activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.BatchNormalization()(x)

    C = y_train.nunique()
    out = keras.layers.Dense(C, activation='softmax')(x)


    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


n_features = len(preprocessor.fit_transform(X_train).T)  


keras_clf = KerasClassifier(
    model=make_deep_model,
    model__input_dim=n_features,           
    model__layers=(512,256,128,64),        
    model__dropout_rate=0.4,                
    model__lr=1e-4,
    optimizer="adam",
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

deep_pipeline = Pipeline([
    ('prep', preprocessor),
    ('nn',  keras_clf)
])


deep_pipeline.fit(X_train, y_train)
print("Deep net test accuracy:", deep_pipeline.score(X_test, y_test) * 100, "%")
 