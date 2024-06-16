from tensorflow.keras.regularizers import l1, l2, l1_l2
from matplotlib_venn import venn2
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.decomposition import PCA
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score


# Loading the dataset
file_path = 'Modified_Aggregated_Ridership_Report.xlsx'
df = pd.read_excel(file_path)

# Feature Engineering
# Specifying the exact format of time data
time_format = '%H:%M:%S'  # Adjust this format to match your data
df['Hour'] = pd.to_datetime(df['Time'], format=time_format).dt.hour
df['Minute'] = pd.to_datetime(df['Time'], format=time_format).dt.minute
df['Total_Minutes'] = df['Hour'] * 60 + df['Minute']
df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
df['Avg_Entries_Per_Bus'] = df['Total_Entries'] / df['Buses_Count']

# Excluding weekends (Saturday and Sunday)
df = df[~df['DayOfWeek'].isin([5, 6])]

# Dropping rows with NaN values in specific columns
df_cleaned = df.dropna(subset=['Total_Minutes', 'Buses_Count'])

# Applying one-hot encoding on 'STOP_LOCATION' & 'DayOfWeek'
onehot_encoder = OneHotEncoder()

# Reapplying one-hot encoding to the cleaned data
encoded_stop_location_cleaned = onehot_encoder.fit_transform(df_cleaned[['STOP_LOCATION', 'DayOfWeek']]).toarray()

# Recombining features for clustering
combined_features_cleaned = np.hstack((encoded_stop_location_cleaned, df_cleaned[['Total_Minutes', 'Buses_Count']]))

# Performing K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(combined_features_cleaned)

# Assigning cluster labels
df_cleaned['Cluster_Label'] = kmeans.labels_

# Analyzing clusters and define class labels
# For example, labelling the clusters based on the median 'Buses_Count' in each cluster
cluster_to_class_map = {
    0: 'Less',
    1: 'Enough',
    2: 'More'
}
df_cleaned['Bus_Class'] = df_cleaned['Cluster_Label'].map(cluster_to_class_map)

# Counting the number of rows in each class
class_counts_cleaned = df_cleaned['Bus_Class'].value_counts()
class_counts_cleaned

# Performing PCA for dimensionality reduction to 2D for visualization purposes
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features_cleaned)

# Now, let's find the top N stop locations by frequency
top_stops = df_cleaned['STOP_LOCATION'].value_counts().head(10).index

# Now filtering the DataFrame to include only these top stops
top_stops_df = df_cleaned[df_cleaned['STOP_LOCATION'].isin(top_stops)]

# Creating a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=top_stops_df, x='Total_Minutes', y='Buses_Count', hue='Cluster_Label', style='STOP_LOCATION')

# Adding titles and labels
plt.title('Bus Count and Time Distribution by Cluster for Top Stops')
plt.xlabel('Total Minutes in the Day')
plt.ylabel('Number of Buses at Stop')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Showing the plot with a tight layout to accommodate the legend
plt.tight_layout()
plt.show()

# One-hot encoding 'STOP_LOCATION' and other categorical features
X = df_cleaned[['STOP_LOCATION', 'Total_Minutes', 'Avg_Entries_Per_Bus', 'DayOfWeek']]
y = df_cleaned['Bus_Class']

# Preprocessing the features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['STOP_LOCATION', 'DayOfWeek']),
        ('num', StandardScaler(), ['Total_Minutes', 'Avg_Entries_Per_Bus'])
    ])
X_processed = preprocessor.fit_transform(X)

# One-hot encoding on the labels
y_encoded = to_categorical(pd.factorize(y)[0])

# Splitting the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42,
                                                  shuffle=True)  # 0.25 x 0.8 = 0.2

# Ensuring your numerical data is in a DataFrame
numerical_features = df_cleaned[['Total_Minutes', 'Buses_Count', 'Avg_Entries_Per_Bus','DayOfWeek']]

# Calculating the correlation matrix
correlation_matrix = numerical_features.corr()

# Displaying the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Defining the model architecture with a function that returns a new model instance
def create_model(input_shape, num_classes, regularization=None, l1_value=0.01, l2_value=0.01):
    if regularization == 'l1':
        reg = l1(l1_value)
    elif regularization == 'l2':
        reg = l2(l2_value)
    elif regularization == 'l1_l2':
        reg = l1_l2(l1=l1_value, l2=l2_value)
    else:
        reg = None  # No regularization

    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape, kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming that X_train.shape[1] gives the number of features and y_encoded.shape[1] is the number of classes
input_shape = (X_train.shape[1],)
num_classes = y_encoded.shape[1]

# Creating three models with different regularization techniques
model_l1 = create_model(input_shape=input_shape, num_classes=num_classes, regularization='l1')
model_l2 = create_model(input_shape=input_shape, num_classes=num_classes, regularization='l2')
model_l1_l2 = create_model(input_shape=input_shape, num_classes=num_classes, regularization='l1_l2')


# Training each model
history_l1 = model_l1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
history_l2 = model_l2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
history_l1_l2 = model_l1_l2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Evaluating each model on the test set
test_loss_l1, test_accuracy_l1 = model_l1.evaluate(X_test, y_test, verbose=0)
test_loss_l2, test_accuracy_l2 = model_l2.evaluate(X_test, y_test, verbose=0)
test_loss_l1_l2, test_accuracy_l1_l2 = model_l1_l2.evaluate(X_test, y_test, verbose=0)

# Displaying the test accuracy for each model
print(f'L1 Model - Test Loss: {test_loss_l1}, Test Accuracy: {test_accuracy_l1}')
print(f'L2 Model - Test Loss: {test_loss_l2}, Test Accuracy: {test_accuracy_l2}')
print(f'L1_L2 Model - Test Loss: {test_loss_l1_l2}, Test Accuracy: {test_accuracy_l1_l2}')

# Incremental training function
def plot_learning_curve_over_examples(model_function, X_train, y_train, X_val, y_val, increments, title):
    training_sizes = [int(len(X_train) * i) for i in increments]
    train_accuracies = []
    val_accuracies = []

    for size in training_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        model = model_function()
        history = model.fit(
            X_train_subset, y_train_subset,
            epochs=100,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
        )

        training_acc = history.history['accuracy'][-1]
        validation_acc = history.history['val_accuracy'][-1]
        train_accuracies.append(training_acc)
        val_accuracies.append(validation_acc)

    plt.figure(figsize=(12, 6))
    plt.plot(training_sizes, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(training_sizes, val_accuracies, 'o-', label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# Choosing increments as fractions of the training data
increments = [0.1, 0.2, 0.5, 0.75, 1.0]

# Defining model functions for L1, L2, and L1_L2 regularization
model_function_l1 = lambda: create_model(input_shape=(X_train.shape[1],), num_classes=y_encoded.shape[1],
                                         regularization='l1')
model_function_l2 = lambda: create_model(input_shape=(X_train.shape[1],), num_classes=y_encoded.shape[1],
                                         regularization='l2')
model_function_l1_l2 = lambda: create_model(input_shape=(X_train.shape[1],), num_classes=y_encoded.shape[1],
                                            regularization='l1_l2')

# Plotting the learning curve for L1
plot_learning_curve_over_examples(model_function_l1, X_train, y_train, X_val, y_val, increments,
                                  'Learning Curve for L1 Regularization')

# Plotting the learning curve for L2
plot_learning_curve_over_examples(model_function_l2, X_train, y_train, X_val, y_val, increments,
                                  'Learning Curve for L2 Regularization')

# Ploting the learning curve for L1_L2
plot_learning_curve_over_examples(model_function_l1_l2, X_train, y_train, X_val, y_val, increments,
                                  'Learning Curve for L1_L2 Regularization')

# Continuing with ROC curve plotting, confusion matrix
def plot_roc_curve(model, X_test, y_test, n_classes, regularization):
    y_pred_probs = model.predict(X_test)
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {regularization} regularization')
    plt.legend(loc="lower right")
    plt.show()

# Defining the function to plot the confusion matrix
def plot_confusion_matrix(model, X_test, y_test, regularization):
    y_pred_classes = np.argmax(model.predict(X_test), axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix for {regularization} regularization')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plotting ROC curves and confusion matrices for each regularization method
n_classes = y_encoded.shape[1]
plot_roc_curve(model_l1, X_test, y_test, n_classes, 'L1')
plot_confusion_matrix(model_l1, X_test, y_test, 'L1')

plot_roc_curve(model_l2, X_test, y_test, n_classes, 'L2')
plot_confusion_matrix(model_l2, X_test, y_test, 'L2')

plot_roc_curve(model_l1_l2, X_test, y_test, n_classes, 'L1_L2')
plot_confusion_matrix(model_l1_l2, X_test, y_test, 'L1_L2')

# Plotting ROC curves and confusion matrices for each regularization method
n_classes = y_encoded.shape[1]

# For the model with L1 regularization
plot_roc_curve(model_l1, X_test, y_test, n_classes, 'L1')
plot_confusion_matrix(model_l1, X_test, y_test, 'L1')

# For the model with L2 regularization
plot_roc_curve(model_l2, X_test, y_test, n_classes, 'L2')
plot_confusion_matrix(model_l2, X_test, y_test, 'L2')

# For the model with both L1 and L2 regularization
plot_roc_curve(model_l1_l2, X_test, y_test, n_classes, 'L1_L2')
plot_confusion_matrix(model_l1_l2, X_test, y_test, 'L1_L2')


# Assuming you have predictions from each model
y_pred_classes_l1 = np.argmax(model_l1.predict(X_test), axis=1)
y_pred_classes_l2 = np.argmax(model_l2.predict(X_test), axis=1)
y_pred_classes_l1_l2 = np.argmax(model_l1_l2.predict(X_test), axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Finding the indices of the errors for each model
errors_l1 = set(np.where(y_pred_classes_l1 != y_true_classes)[0])
errors_l2 = set(np.where(y_pred_classes_l2 != y_true_classes)[0])
errors_l1_l2 = set(np.where(y_pred_classes_l1_l2 != y_true_classes)[0])

# Calculating precision and recall for L1 model
precision_l1 = precision_score(y_true_classes, y_pred_classes_l1, average='weighted')
recall_l1 = recall_score(y_true_classes, y_pred_classes_l1, average='weighted')
print(f'L1 Model - Precision: {precision_l1}, Recall: {recall_l1}')

# Calculating precision and recall for L2 model
precision_l2 = precision_score(y_true_classes, y_pred_classes_l2, average='weighted')
recall_l2 = recall_score(y_true_classes, y_pred_classes_l2, average='weighted')
print(f'L2 Model - Precision: {precision_l2}, Recall: {recall_l2}')

# Calculating precision and recall for L1_L2 model
precision_l1_l2 = precision_score(y_true_classes, y_pred_classes_l1_l2, average='weighted')
recall_l1_l2 = recall_score(y_true_classes, y_pred_classes_l1_l2, average='weighted')
print(f'L1_L2 Model - Precision: {precision_l1_l2}, Recall: {recall_l1_l2}')

# Venn Diagrams
plt.figure(figsize=(12, 12))
venn2(subsets=(len(errors_l1), len(errors_l2), len(errors_l1.intersection(errors_l2))),
      set_labels=('L1 Errors', 'L2 Errors'))
plt.title('Venn Diagram of Errors between L1 and L2 Models')
plt.show()

plt.figure(figsize=(12, 12))
venn2(subsets=(len(errors_l1), len(errors_l1_l2), len(errors_l1.intersection(errors_l1_l2))),
      set_labels=('L1 Errors', 'L1_L2 Errors'))
plt.title('Venn Diagram of Errors between L1 and L1_L2 Models')
plt.show()

plt.figure(figsize=(12, 12))
venn2(subsets=(len(errors_l2), len(errors_l1_l2), len(errors_l2.intersection(errors_l1_l2))),
      set_labels=('L2 Errors', 'L1_L2 Errors'))
plt.title('Venn Diagram of Errors between L2 and L1_L2 Models')
plt.show()

# Analyzing Sample Errors
def analyze_sample_errors(errors_set, X_data, y_pred, y_true, title):
    sample_indices = np.random.choice(list(errors_set), size=min(5, len(errors_set)), replace=False)
    print(f"Sample Errors for {title}:")
    for index in sample_indices:
        print(f"Index: {index}, True label: {y_true[index]}, Predicted label: {y_pred[index]}")
        # Optionally display the feature values or a plot if X_data is image data
        # plt.imshow(X_data[index]) # for image data
        # plt.show()

# Analyzing Sample Errors for each model
analyze_sample_errors(errors_l1, X_test, y_pred_classes_l1, y_true_classes, "L1 Model")
analyze_sample_errors(errors_l2, X_test, y_pred_classes_l2, y_true_classes, "L2 Model")
analyze_sample_errors(errors_l1_l2, X_test, y_pred_classes_l1_l2, y_true_classes, "L1_L2 Model")


