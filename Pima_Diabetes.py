import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")

classe_0 = df[df["Outcome"] == 0].values.tolist()
classe_1 = df[df["Outcome"] == 1].values.tolist()

while len(classe_1) < len(classe_0):
    classe_1.append(random.choice(classe_1))

dados_balanceados = classe_0 + classe_1

random.shuffle(dados_balanceados)

df_balanceado = pd.DataFrame(dados_balanceados, columns=df.columns)

X = df_balanceado.drop("Outcome", axis=1)
y = df_balanceado["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

svm = SVC(
    C=1,
    kernel="rbf",
    gamma="scale"
)

knn = KNeighborsClassifier(
    n_neighbors=7
)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("===== RESULTADOS =====")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"KNN Accuracy: {knn_acc:.4f}")

resultados = {
    "Random Forest": rf_acc,
    "SVM": svm_acc,
    "KNN": knn_acc
}

melhor_modelo = max(resultados, key=resultados.get)

print(f"\nMelhor modelo para produção: {melhor_modelo}")