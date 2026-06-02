import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

ros = RandomOverSampler(random_state=42)
X_train_balanceado, y_train_balanceado = ros.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanceado)
X_test_scaled = scaler.transform(X_test)


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

rf.fit(X_train_scaled, y_train_balanceado)
svm.fit(X_train_scaled, y_train_balanceado)
knn.fit(X_train_scaled, y_train_balanceado)

rf_pred = rf.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)
knn_pred = knn.predict(X_test_scaled)

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

print("\n===== COMPARAÇÃO DOS MODELOS =====")
print(f"Random Forest ({rf_acc:.4f})")
print(f"SVM ({svm_acc:.4f})")
print(f"KNN ({knn_acc:.4f})")

print("\n===== RELATÓRIOS DE CLASSIFICAÇÃO =====")

print("\n--- RANDOM FOREST ---")
print(classification_report(y_test, rf_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, rf_pred))

print("\n--- SVM ---")
print(classification_report(y_test, svm_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, svm_pred))

print("\n--- KNN ---")
print(classification_report(y_test, knn_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, knn_pred))

print("\n===== CONCLUSÃO =====")
print(f"O modelo com maior acurácia foi: {melhor_modelo}")

if melhor_modelo == "Random Forest":
    print(
        "O Random Forest foi escolhido para produção por apresentar "
        "a melhor acurácia e boa capacidade de generalização."
    )
elif melhor_modelo == "SVM":
    print(
        "O SVM foi escolhido para produção por apresentar "
        "o melhor desempenho entre os modelos avaliados."
    )
else:
    print(
        "O KNN foi escolhido para produção por apresentar "
        "a maior acurácia neste conjunto de dados."
    )