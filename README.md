This project's script uses Google Cloud BigQuery, PySpark, scikit-learn, seaborn, matplotlib and Plotly to analyze and visualize a dataset regarding mobile network coverage in Catalonia between 2015 and 2017.

# Features
 *   Big Data Integration: Uses BigQuery to access and query large datasets directly.
 *   Data Cleaning and Transformation: Handles missing values and normalizes features using PySpark.
 *   Correlation Analysis: Provides a heatmap of correlations between key features.
 *   Regression Models: Implements Logistic Regression, Decision Tree, Random Forest, and K-Means Clustering to predict average signal values of the internet and compare them to the internet speed values and number of satellites.
 *   Evaluation: Calculates performance metrics for each model.

# Install and import libraries
Install the required libraries (google-cloud-bigquery, pyspark, matplotlib, scikit-learn) and import the modules for data analysis, graph visualization (seaborn, Plotly), and machine learning (e. g. Logistic Regression, Decision Tree). It's also necessary to set up a Google Cloud project and enable the BigQuery API.

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
* Compare the actual values ​​vs. predicted values ​​in a Plotly graph.

3. Clustering with K-Means
* Apply K-Means Clustering with k=3 groups.
* Performance is evaluated using Silhouette Score.
* Visualize clusters in an interactive 3D scatter plot.

# Conclusion
This script automates the entire process of:
* Collecting data from BigQuery.
* Data cleaning and preprocessing.
* Statistical analysis and visualization.
* Building and evaluating ML models.
* It uses both PySpark (for scalability) and Pandas & Plotly (for detailed visualizations). This type of analysis is useful for telecommunications, mobile network optimization, and understanding the impact of environmental factors on mobile internet performance.

# License


# Contribution
