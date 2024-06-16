<!-- ###

README.md
===========

Title: Predictive Analysis of Bus Ridership Occupancy Levels

Description:
This project aims to classify the adequacy of bus services at different times and stops into three categories: Not Enough, Enough, and More Than Required. The neural network models implemented attempt to predict the bus service adequacy based on historical ridership data and time-of-day bus frequencies.

Folder Structure:
/Term-Project-Shreyas-Shirsekar
    /Code
        nn_model_revised3.py - Main script for running the neural network models.

How to Run the Code:

1. Ensure that Python 3.6 or above is installed on your system.

2. Install the required libraries mentioned in the 'requirements.txt' file using the command:
   pip install -r requirements.txt

3. Navigate to the 'Code' directory in the terminal or command prompt.

4. Run the script using the command:
   python3 nn_model_revised3.py

5. Follow any on-screen prompts and close the graphs that pop up to move on to the next part of the code.

Data:
The dataset used for the model is in the form of an Excel file 'Modified_Aggregated_Ridership_Report.xlsx'. This dataset must be placed in the same directory as the script or in a data-specific directory which the script references. Make changes to the path in the code accordingly.

Code Organization:
The code for Predictive Analysis of Bus Ridership Occupancy Levels project is structured into several distinct sections, each playing a crucial role in the overall machine learning pipeline. Here's a breakdown of the code structure:

1. Importing Libraries:
   The script begins by importing necessary Python libraries such as TensorFlow, scikit-learn, pandas, seaborn, numpy, etc., which are essential for data processing, model building, and visualization.

2. Data Loading:
   The dataset, 'Modified_Aggregated_Ridership_Report.xlsx', is loaded using pandas. This step is crucial for accessing the data required for training and testing the models.

3. Feature Engineering:
   This section includes the transformation and creation of new features from the raw data, such as extracting hours and minutes from time data and calculating average entries per bus. It prepares the dataset for more effective model training.

4. Data Preprocessing:
   The script applies preprocessing techniques like one-hot encoding for categorical variables and standard scaling for numerical features. This standardizes the data, making it suitable for neural network processing.

5. Clustering and PCA:
   K-Means clustering is performed to segment the data, and PCA is used for dimensionality reduction, primarily for visualization purposes.

6. Model Building:
   The core of the script involves defining and creating the neural network models. It includes setting up a Sequential model with layers like Dense, Dropout, and BatchNormalization. Regularization models (L1, L2, and L1_L2) are also integrated here.

7. Model Training and Validation:
   Each model is trained and validated on the dataset. The script uses EarlyStopping for more efficient training.

8. Model Evaluation:
   After training, the models are evaluated using the test set. Metrics like precision and recall are calculated, and results are printed.

9. Visualization:
   The script generates various plots such as ROC curves, confusion matrices, learning curves, and Venn diagrams to visualize the model's performance and error analysis.

10. Error Analysis:
   A dedicated section for analyzing sample errors is included, providing insights into the model's predictions.

11. Utility Functions:
   Functions for plotting learning curves, ROC curves, confusion matrices, and analyzing errors are defined and used throughout the script.

Each section is marked with comments for easy identification and understanding. The script is designed to be executed sequentially, with each part building upon the previous steps.



Output:
The script outputs visualizations such as ROC curves, confusion matrices, and learning curves, which are displayed on execution. Precision and recall metrics are printed to the console. 

Notes:
- Modify any file paths in the script if your data or output directories are different from the provided structure.
- This project is designed to run on a single machine and is not configured for distributed computing environments.

Contact:
For any queries regarding the project or code, please contact Shreyas Shirsekar at s.shirsekar001@umb.edu

Acknowledgments:
A special thanks to UMass Boston Transportation Department and Paul Revere Transportation for providing the data and the opportunity to work on this meaningful project.
### -->