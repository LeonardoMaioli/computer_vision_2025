import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Carregar CSV
df = pd.read_csv('dados_gesto.csv')  # Substitua pelo caminho do seu arquivo



# Separar features e rótulos
X = df.drop('label', axis=1).values
y = df['label'].values

# Codificar rótulos (se forem strings)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Salvar modelo e codificador
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
