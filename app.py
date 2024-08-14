import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
def loading_Data():
    # Load the dataset
    df = pd.read_csv('customer_churn.csv')
    print(df.head())

    """
    Data Exploration

    Check for missing values and understand the data types of each column.
    Use describe() to get a summary of numerical features.
    """
    print(df.info())
    print(df.describe())
    """
    Handling Dates

    Convert the Onboard_date to a datetime object and create new features if necessary, 
    such as the number of days since onboarding.
    """
    df['Onboard_date'] = pd.to_datetime(df['Onboard_date'])
    df['Days_since_onboard'] = (pd.to_datetime('today') - df['Onboard_date']).dt.days
    return df

def Data_Preprocessing(df):
    """
    Step 2: Data Cleaning and Preprocessing
    Drop Unnecessary Columns
    Columns like Name, Company, and Location may not be useful for prediction. 
    You can choose to drop them unless you want to perform additional analysis.
    """
    df = df.drop(columns=['Names', 'Company', 'Location', 'Onboard_date'])
    """Feature Engineering
    Create new features based on existing ones, such as Ads_per_year = Total_Purchase / Years."""
    df['Ads_per_year'] = df['Total_Purchase'] / df['Years']
    """
     Handling Categorical Variables
    Since most features are numerical or binary, you might only need to encode 
    the Account_Manager column if it is not already numeric.
    """
    df['Account_Manager'] = df['Account_Manager'].astype(int)
    """
    Normalization/Scaling
    Scale the numerical features to ensure they are on a similar scale, which helps 
    some machine learning algorithms perform better."""

    scaler = StandardScaler()
    df[['Age', 'Total_Purchase', 'Years', 'Num_sites', 'Days_since_onboard', 'Ads_per_year']] = scaler.fit_transform(df[['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Days_since_onboard', 'Ads_per_year']])

    return df

def EDA_fun(df):
    """Visualize the Data
    Use Seaborn or Matplotlib to create visualizations that help understand the distribution 
    of features and relationships between them."""


    # Distribution of the target variable (Churn or Not)
    sns.countplot(x='Churn', data=df)
    plt.show()

    # Pairplot to see relationships between features
    sns.pairplot(df, hue='Churn')
    plt.show()
    """Correlation Matrix
        Check the correlation between features to understand which variables 
        might be the most influential in predicting churn.
    """
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def ModelBuilding(X_train, X_test, y_train, y_test):

    """
    Train the Model

    Choose a machine learning model, such as Logistic Regression, 
    Random Forest, or Gradient Boosting. Train the model on the training data.
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    """Evaluate the Model
        Evaluate the model using the test set and print metrics 
        such as accuracy, precision, recall, and F1-score.
    """
    

    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def ModelEvaluation(model, X, y):
    
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores: ", cv_scores)
    print("Mean Score: ", cv_scores.mean())

def main():

    df= loading_Data()
    df=Data_Preprocessing(df)
    #Step 3: Exploratory Data Analysis (EDA)
    #EDA_fun(df)
    """
    Split the Data
    Separate the dataset into features (X) and target (y) variables,
    and then split into training and testing sets.
    """
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #Step 4: Model Building
    model=ModelBuilding(X_train, X_test, y_train, y_test)

    """
    Step 5: Model Evaluation and Tuning
    Cross-Validation
    Perform cross-validation to ensure the model's robustness.
    """
    ModelEvaluation(model, X, y)

    

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    print("Best Parameters: ", grid_search.best_params_)


if __name__=="__main__":
    main()