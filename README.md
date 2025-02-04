This project's script uses Google Cloud BigQuery, PySpark, scikit-learn, seaborn, matplotlib and Plotly to analyze and visualize a dataset regarding mobile network coverage in Catalonia between 2015 and 2017.

# Features
 *   Big Data Integration: Uses BigQuery to access and query large datasets directly.
 *   Data Cleaning and Transformation: Handles missing values and normalizes features using PySpark.
 *   Correlation Analysis: Provides a heatmap of correlations between key features.
 *   Regression Models: Implements Logistic Regression, Decision Tree, Random Forest, and K-Means Clustering to predict average signal values of the internet and compare them to the internet speed values and number of satellites.
 *   Evaluation: Calculates performance metrics for each model.

# Install and import libraries
Install the required libraries (google-cloud-bigquery, pyspark, matplotlib, scikit-learn) and import the modules for data analysis, graph visualization (seaborn, Plotly), and machine learning (e. g. Logistic Regression, Decision Tree). It's also necessary to set up a Google Cloud project and enable the BigQuery API.


# Usage
1. Clone the repository using the web URL:
   ```bash
   https://github.com/stteo150/ProiectLA.git
   ```
2. Open the LAPr.zip archive and extract the file with the extension .ipynb
3. Open the file in Google Colab
4. Configure your BigQuery credentials and project details in Google Colab.
5. Execute the cells sequentially to analyze the data and generate predictions.

# Dataset
This project utilizes a **public Google BigQuery dataset** that includes:
* Average internet signal
* Network type and name
* Average net speed
* Number of GPS satellites
* Mobile operator name
* precison - the provider's accuracy
* Position provider name

# Big Data Technologies
* Google BigQuery: Enables querying of large datasets efficiently.
* PySpark: Facilitates distributed data processing and machine learning.

# Authenticate and query BigQuery
* Specify a JSON file for authentication to Google Cloud.
* Create a BigQuery client to run a SQL query.
* Query a public database of mobile coverage and select relevant columns (e.g. date, hour, signal, network, speed).
* Limit the results to a certain number of rows and convert them to a pandas.DataFrame.

# Clean the data
* Remove null values.
* Convert numeric columns (signal, speed, satellites, precision) to float.

# Convert to PySpark DataFrame
* Create a PySpark session.
* Define the schema for the data (StructType).
* Load pandas.DataFrame into a pyspark.DataFrame and remove null values ​​again.

# Explore the dataset with PySpark
* Display the schema of the DataFrame.
* Count the total rows.
* Filter and display only rows where signal > 50.
* Calculate the average and standard deviation of speed.
* Count the occurrences of each network type.
* Create a new column speed_kbps (speed converted to kbps).
* Sort and display the highest speeds.

# Save data to a CSV file
* The processed data is saved in an output.csv file for later use.

# Analysis of correlations between variables
* Calculate Pearson correlation coefficients between signal, speed, satellites, precision.
* Construct a correlation matrix.
* Display a heatmap with Seaborn to visualize the correlations.

# Interactive graphical visualizations
* Bar Chart: signal distribution.
* Line Chart: average speed (speed) as a function of time (hour).
* Scatter Plot: correlation between signal and speed.
* Line Chart (Plotly): speed evolution over time.
* Pie Chart: distribution of mobile networks.
* Bubble Chart: relationship between signal, speed and number of satellites.
* 3D Scatter Plot: analysis of the relationship between signal, speed and precision.
* Histogram: distribution of precision and speed.
* Sunburst Chart: hierarchy of operators, networks and activities.

# Machine Learning (ML) Models
Apply classification and clustering models using PySpark ML:
* Create a VectorAssembler to combine variables (signal, speed, satellites, precision).
* Split the dataset into 80% training data and 20% test data.

1. Logistic Regression
* Apply Logistic Regression to predict the signal.
* Evaluate the accuracy of the model using MulticlassClassificationEvaluator.
* Visualize the actual values ​​vs. predicted values ​​in a Plotly graph.

2. Decision Tree & Random Forest
* Apply a decision tree (DecisionTreeClassifier) ​​and a Random Forest (RandomForestClassifier).
* Evaluate the accuracy of each model.
* Compare the actual values ​​vs. predicted values of internet signal ​​in a Plotly graph.

3. Clustering with K-Means
* Apply K-Means Clustering with k=3 groups for net speed, signal and GPS satellites.
* Performance is evaluated using Silhouette Score.
* Visualize clusters in an interactive 3D scatter plot.

# Results
1. Logistic Regression Accuracy
* Accuracy of the Logistic Regression model: 89.7 %
* Interpretation: The Logistic Regression model performed best among all the classification models. An accuracy of 89.7% indicates that the model is able to correctly classify the signal (signal label) based on the features.
* Advantages: Logistic Regression is a simple and interpretable algorithm that works well in cases where the data is linearly separable.
* Possible improvements: If the data is not completely linearly separable, additional preprocessing or the use of a more complex model may be required.

2. K-Means Clustering Silhouette Score
* Silhouette Score: 73.6 %
* Interpretation: The Silhouette Score is a measure of the cohesion and separation of clusters. A score of 73.6% indicates a good quality of the clusters formed by K-Means. The data is well grouped according to the provided features (signal, speed, satellites, precision).
* Advantages: K-Means is an efficient algorithm for clustering, and a high score suggests that the data is well distributed in the feature space.
* Possible improvements: Increasing the number of clusters (k) could improve the results if there are more distinct groups in the data.

3. Decision Tree Classifier Accuracy
* Accuracy of the Decision Tree model: 55.6 %
* Interpretation: The Decision Tree model performs significantly worse, with an accuracy of only 55.6%. This may indicate that the model has overestimated the noise in the data or that there are complex relationships between features that a single decision tree cannot capture.
* Advantages: The decision tree is easy to interpret and can highlight simple rules in the data.
* Disadvantages: It tends to overfit in the case of a  highly variable data set.
* Possible improvements: Use regularization techniques or switch to a more robust algorithm, such as Random Forest.

4. Random Forest Classifier Accuracy
* Random Forest Model Accuracy: 56.3 %
* Interpretation: Although Random Forest is a more robust extension of Decision Tree, the accuracy is slightly higher than Decision Tree, at 56.3%. This result may be influenced by the fact that the data is not complex enough or the number of trees is too small.
* Advantages: Random Forest is more robust and reduces the overlap of noise in the data.
* Disadvantages: May require careful configuration of hyperparameters (e.g., number of trees or maximum depth).
* Possible improvements: Tuning hyperparameters and using more balanced datasets can increase accuracy.

# Conclusion
This script automates the entire process of:
* Collecting data from BigQuery.
* Data cleaning and preprocessing.
* Statistical analysis and visualization.
* Building and evaluating ML models.
* It uses both PySpark (for scalability) and Pandas & Plotly (for detailed visualizations). This type of analysis is useful for telecommunications, mobile network optimization, and understanding the impact of environmental factors on mobile internet performance.
* Logistic Regression had the best performance (89.7%), suggesting that the data is well represented by a linear model.
* K-Means formed good quality clusters (Silhouette Score: 0.736), indicating a clear structure in the data.
* Decision Tree and Random Forest models perform similarly but are inferior to Logistic Regression. They could be improved by optimizing parameters.

# License
This project is open-source and available under the [GNU General Public License v3.0](LICENSE)

# Contribution
Feel free to contribute to this project by creating issues or submitting pull requests.
