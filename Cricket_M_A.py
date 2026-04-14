import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier # Upgraded from DecisionTree
from sklearn.metrics import accuracy_score

# 1. Load & Clean
df = pd.read_csv("ODI_Match_info.csv")
df = df.dropna(subset=['winner'])

# 2. Advanced Feature Engineering
# Create a feature for whether the home team (team1) won the toss
df['team1_toss_win'] = (df['toss_winner'] == df['team1']).astype(int)

# Fill missing cities based on venue (mapping the most common city for a venue)
venue_city_map = df.groupby('venue')['city'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
df['city'] = df.apply(lambda row: venue_city_map[row['venue']] if pd.isnull(row['city']) else row['city'], axis=1)

# 3. Features & Target
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city', 'team1_toss_win']
X = df[features]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline with Random Forest
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), 
                   ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city'])],
    remainder='passthrough'
)

# Use Random Forest for better accuracy
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# 5. Hyperparameter Tuning
param_grid = {
    'classifier__max_depth': [20, 30, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Improved Accuracy: {accuracy_score(y_test, best_model.predict(X_test)):.2f}")

# 6. Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# print("Accuracy :", accuracy_score(y_test, ))






