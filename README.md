# Session-Aware Intent & Anomaly Detection for Enterprise AI Query Telemetry
## Setup Instructions
1. Clone this repository
2. Use pip to install the following modules:
   - jupyter
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - sklearn
   - keras
   - plotly

or use ```pip install -r requirements.txt```
3. Run ```jupyter notebook notebooks``` to read both notebooks.

## Architecture and Pipeline Overview

### Support Vector Machine
The Support Vector Machine pipeline is layed out as follows:

#### Preprocessing
The preprocessing step of the pipeline consists of one hot coding ```declared_intent```, ```user_role```, and ```enterprise_id``` so that they could be processed in the SVM as the first step. The process then splits the dataset into 19 features and values. The 19 features are as follows:
```
intent_drift_score
unique_tables_accessed
entropy_table_acces_sequence
duration_seconds
sensitive_table_access_count
row_estimate
num_events
cross_table_join_count
time_delta_mean
confidence_mean
user_role
declared_intent
select_count
update_count
join_count
aggregation_count
ddl_count
auth_count
enterprise_id
```
These features are normalized to z-scores by sklearn's StandardScaler and are reduced to 2 PCA components to reduce overfitting. The dataset is then split even further into training, testing, and validation splits randomly on a 70/15/15 percent split.

#### Cross Validation
The SVM classifier is setup in this step with ```class_weights``` set to balanced. This function decides the best kernel and the best C value for the SVM. The best parameters are returned and fit with the training data.

#### Testing and Metrics
We use our test values to make predictions on session behaviour. The output of the classifier is a binary classification where 0 indicates normal behaviour while 1 indicates suspicious behaviour. These predictions are evaluating using confusion matrices, F1 score, precision, recall, and AUROC curves. For SVMs, we also evaluate the decision boundary to see where the function is dividing our values.

### Long Short-Term Memory Model
The architecture of the LSTM model is constructed as follows: First, the class weights are balanced based on the original training data by sklearn and stored in a dictionary. Second, the features are inputted into the model in batches of 32. These features then are processed by the LSTM layer to train for sequences. These batches then go through the batch normalization and dropout layers to prevent overfitting. The next layer is a rectified linear unit layer that is smaller than the LSTM layer and turns all negative values from LSTM into postitive values. After going through one more dropout and batch normalization layer, the outputs are generated with a sigmoid layer to produce a binary output of 0 or 1. The model evaluates performance by using binary cross-entropy loss and optimizes performance by using adaptive moment estimation (Adam) and by using accuracy as a metric. The model uses a default learning rate of 0.01 and trains for 100 epochs.

The pipeline for LSTM is as follows:

#### Preprocessing
The first step of preprocessing is that the start and end times are converted to Unix seconds to comply with LSTM processing. The duration is then calculated as the difference between the start and end times. Next, the sequences are split by order in a 70/15/15 percent train/validation/test split and split by features and labels. The six features of LSTM anomaly detection are as follows:
```
intent_drift_score
unique_tables_accessed
entropy_table_access_sequence
duration_seconds
sensitive_table_access_count
row_estimate
```
These features are then normalized to z-scores in the same way as SVM preprocessing before being reshaped to fit LSTM by adding a third dimension of 1 into all datasets.

### Training and Testing
The model is fit using training, testing, and validation data. The model is fit over validation data through 100 epochs to improve performance of the model. We use validation loss, precision, recall, F1 score, AUROC curves, and confusion matrices to keep track of the data. We also test the outputs with a confidence threshold of 0.5 since LSTM would not give us a concrete value for each function.

## Assumptions and Design Decisions

### Assumptions
For both SVM and LSTM, our intent drift score calculations assume that both the declared intent of the session and the user's role matter in calculations. For LSTM, we assume that the chronological order of which the sessions occured matter to predicting anomalies.

### Intent Drift Score
The intent drift score is calculated by the equation below:

$F(x) = \sum_{k=1}^n x_1 * x_2 * x_3 * x_4 * log_2(r) / n$, where $x_1$ is the user role multiplier, $x_2$ is the intent multiplier, $x_3$ is the intent mismatch multiplier, $x_4$ is the restricted access multiplier, $r$ is the estimate of the number of rows in a query, and $n$ is the number of events in a session.

$x_1$ is either 1 or 2 depending on both the user's role and the table accessed. The tables with a multiplier of 2 are as follows:

- Executive and Admin: None
- Analyst, Tier 1 Support, and Support: Employee information (employee_salary, executive_compensation), administrator tables (audit_log, system_credentials), and private information (customer_pii, users)
- Data Engineer: Customer Information (customer_pii) and system_credentials

$x_2$ is either 1 or 2 depending on both the declared intent and the query type. All intents multiply this value by 2 with data definition language (CREATE, DROP, ALTER TABLE, etc.) and AUTH queries, but billing summaries, customer lookups, and compensation reviews multiply risk by 2 for UPDATE queries.

$x_3$ is 1 when the intention of the query matches the declared intention of the session and 2 otherwise.

$x_4$ is 2 when the outcome of a query is a 403 Forbidden code and 1 otherwise.

$r$ is the number of rows in a query to attempt to root out broad retrievals and table scans.

$n$ divides the original score by the number of queries to make the threshold of anomalies easier to read and to avoid gaps of 10s-100s between sessions.

### Model Choice

We decided on Support Vector Machine as a baseline because the model is strong with high dimensional data. This feature of SVMs will give a strong performance for LSTMs to compare against as well as checking if less features outperforms more features in modeling anomaly detection. 

We decided on LSTMs as the sequential model because of how the model predicts based on time series. The start and end times feature in LSTM classifiers give us ordering for every session, which gives the dataset the ability to be analysed in the context of time.

## Future Improvements
While our LSTM model performed well, one aspect that we could improve in terms of our intent drift score is the calculation of both broad retrieval and table scan sessions and off-hour sensitive access or high volume access performance. In both pattern labels, the intent drift score clustered the anomalies close to benign and slow drip exfiltration sessions. Refining the formula to better seperate both classes will result in better performances for LSTM. A method that could be tested includes revising the calculations involving the number of rows to be more significant after a row estimate threshold.

Another significant improvement the LSTM could make is decreasing the number of epochs for the model to arrive at the best performance. This could be done by changing the optimized from Adam to AdamW or revising the metric for the LSTM to a classification metric like AUC or recall. This will result in quicker training times for the model.
