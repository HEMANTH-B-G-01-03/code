import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
# Manually entering a small dataset
data = {
 "age": [29, 45, 34, 60, 50, 41, 52, 39, 48, 59],
 "cholesterol": [200, 240, 210, 280, 260, 230, 300, 220, 250, 270],
 "bp": [120, 140, 130, 150, 140, 135, 160, 125, 145, 155],
 "heart_disease": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
}
# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["age", "cholesterol", "bp"]]
y = df["heart_disease"]
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
# Predictions
y_pred = nb_model.predict(X_test)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Visualization of Results
plt.scatter(df["cholesterol"], df["bp"], c=df["heart_disease"], cmap="coolwarm", label="DataPoints")
plt.xlabel("Cholesterol")
plt.ylabel("Blood Pressure")
plt.title("Heart Disease Diagnosis")
plt.legend(["No Disease", "Disease"], loc="best")
plt.show()
# Bayesian Network Visualization
G = nx.DiGraph()
G.add_edges_from([
 ("age", "heart_disease"),
 ("cholesterol", "heart_disease"),
 ("bp", "heart_disease")
])
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=3000, node_color="lightblue", font_size=10,
font_weight="bold")
plt.title("Bayesian Network for Heart Disease Diagnosis")
plt.show() 