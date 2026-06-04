import pandas as pd
import random
import joblib
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
    X, y, test_size=0.2, random_state=42
)

ros = RandomOverSampler(random_state=42)
X_train_balanceado, y_train_balanceado = ros.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanceado)
X_test_scaled = scaler.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
svm = SVC(C=1, kernel="rbf", gamma="scale")
knn = KNeighborsClassifier(n_neighbors=7)

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
nome_melhor_modelo = max(resultados, key=resultados.get)
acuracia_melhor_modelo = resultados[nome_melhor_modelo]

modelos_objetos = {
    "Random Forest": rf,
    "SVM": svm,
    "KNN": knn
}
modelo_campeao = modelos_objetos[nome_melhor_modelo]

print("\n===== MODELO SELECIONADO =====")
print(f"Método Estimador Vencedor: {modelo_campeao.__class__.__name__} ({nome_melhor_modelo})")
print(f"Acurácia do Campeão: {acuracia_melhor_modelo:.4f}")

joblib.dump(modelo_campeao, 'melhor_modelo_diabetes.pkl')
joblib.dump(scaler, 'scaler_diabetes.pkl')

def modulo_de_inferencia(novos_dados_dict):
    modelo_carregado = joblib.load('melhor_modelo_diabetes.pkl')
    scaler_carregado = joblib.load('scaler_diabetes.pkl')
    
    df_novo = pd.DataFrame([novos_dados_dict])
    X_novo_escalonado = scaler_carregado.transform(df_novo)
    previsao_numerica = modelo_carregado.predict(X_novo_escalonado)[0]
    
    return "Positivo para Diabetes" if previsao_numerica == 1 else "Negativo para Diabetes"

paciente_exemplo = {
    'Pregnancies': 2,
    'Glucose': 130,
    'BloodPressure': 70,
    'SkinThickness': 28,
    'Insulin': 115,
    'BMI': 32.4,
    'DiabetesPedigreeFunction': 0.4,
    'Age': 35
}

resultado = modulo_de_inferencia(paciente_exemplo)
print("\n===== TESTE DO MÓDULO DE INFERÊNCIA =====")
print(f"Dados do Paciente: {paciente_exemplo}")
print(f">> PREVISÃO FINAL: {resultado}")