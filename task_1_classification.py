import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Step 1: Load the dataset manually
try:
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Training%20Dataset.arff"
    data = pd.read_csv(data_url, header=None)  # You might need to adjust column names based on the dataset
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    raise

# Step 2: Prepare features and target
try:
    column_names = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "double_slash_redirecting",
                    "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
                    "Favicon", "port", "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags",
                    "SFH", "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover", "RightClick",
                    "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
                    "Google_Index", "Links_pointing_to_page", "Statistical_report", "Result"]  # Adjust column names accordingly

    data.columns = column_names  # Set appropriate column names

    # Debug: Display data structure
    print("Combined DataFrame Shape:", data.shape)
    print("Combined DataFrame Head:\n", data.head())

    # Separate features and target
    X = data.drop("Result", axis=1)  # Features
    y = data["Result"]  # Target

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

except KeyError as e:
    print(f"Dataset structure is unexpected. Missing key: {e}")
    raise
except Exception as e:
    print(f"Error preparing data: {e}")
    raise

# Step 4: Model 1 - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 5: Model 2 - Support Vector Machine
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Step 6: Evaluation
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrices
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()

# Step 7: Multiclass ROC Curve for Random Forest
classes = sorted(y.unique())  # Get all unique class labels
y_test_bin = label_binarize(y_test, classes=classes)  # Convert y_test to binary format for each class
n_classes = len(classes)

# Compute ROC curve and ROC area for each class
rf_probs = rf_model.predict_proba(X_test)  # Predict probabilities for Random Forest
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])  # ROC for class `i`
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.title("Random Forest ROC Curve (Multiclass)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
