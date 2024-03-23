from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.backends.backend_tkagg as tkagg
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#
def load_data():
    data = pd.read_csv('all_w10_s1.csv')

    # 数据随机采样一�?
    data = data.sample(frac=0.01, random_state=42)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1].str.split('_', expand=True)[0]
    y = pd.Series(y)

    # 把y的标签转换成数字
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X,y

def train(X, y, classifier):
    model_select = {
        "Perceptron": Perceptron(),
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Boosting": GradientBoostingClassifier(),
        "Multilayer Perceptron": MLPClassifier()
    }
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('preprocessor', PolynomialFeatures()),
        ('classifier', model_select[classifier])
    ])
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Predict the labels for the test data
    y_pred = pipeline.predict(X_test)
    # Print the classification report
    cp = classification_report(y_test, y_pred)
    # print("Classification Report:")
    # print(cp)
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    return cp, cm

def grid_searching(X,y,classifier):
    # 划分训练集和测试�?
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('preprocessor', PolynomialFeatures()),
        ('classifier', Perceptron())
    ])

    # Define the parameter grid
    param_grids = [
        {
            'preprocessor': [PolynomialFeatures()],
            'preprocessor__degree': [2, 3, 4]
        },
        {
            'preprocessor': [PCA()],
            'preprocessor__n_components': [2, 3, 4]
        },
        {
            'preprocessor': [LinearDiscriminantAnalysis()],
            'preprocessor__solver': ['svd', 'eigen'],
            'preprocessor__tol': [0.0001, 0.001, 0.01],
            'preprocessor__store_covariance': [True, False]
        },
        {
            'preprocessor': [QuadraticDiscriminantAnalysis()],
            'preprocessor__reg_param': [0.0, 0.1, 0.2],
            'preprocessor__tol': [0.0001, 0.001, 0.01]
        }
    ]

    model_param_grids = {
        "Perceptron":{
            'classifier': [Perceptron()],
            'classifier__max_iter': [50, 100, 200],
            'classifier__tol': [0.001, 0.0001]
        },
        "Logistic Regression":{
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__max_iter': [100, 200, 300]
        },
        "SVM":{
            'classifier': [SVC()],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        },
        "Decision Tree":{
            'classifier': [DecisionTreeClassifier()],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "Random Forest":{
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [10, 50, 100],
            'classifier__max_depth': [None, 5, 10]
        },
        "Boosting":{
            'classifier': [GradientBoostingClassifier()],
            'classifier__n_estimators': [10, 50, 100],
            'classifier__learning_rate': [0.01, 0.1, 1.0]
        },
        "Multilayer Perceptron":{
            'classifier': [MLPClassifier()],
            'classifier__hidden_layer_sizes': [(50,), (100,)],
            'classifier__activation': ['tanh', 'relu'],
            'classifier__max_iter': [200, 300]
        }
    }

    # Get the classifier parameters from the model_param_grids
    classifier_params = model_param_grids[classifier]

    # Create the parameter grid
    param_grids = [{**pg, **classifier_params} for pg in param_grids]

    print(param_grids)
    
    #Perceptron, Logistic Regression, SVM, Decision Tree, Random Forest, Boosting, and Multilayer Perceptron.
    # 在这里用您的数据集训�? pipeline# 在这里用您的数据集训�?
    grid_search = GridSearchCV(pipeline, param_grids, cv=5,n_jobs=-1)

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and preprocessing options
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best hyperparameters and preprocessing options
    print("Best Hyperparameters and Preprocessing Options:")
    print(best_params)
    print("Best Score:")
    print(best_score)
    
    return best_params, best_score

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classifier UI")
        self.geometry("1000x800")
        
        self.classifier_var = tk.StringVar()
        self.classifier_var.set("Perceptron")
        
        self.steps_var = tk.StringVar()
        self.steps_var.set("Train and Analyze")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Classifier selection
        classifier_label = ttk.Label(self, text="Select Classifier:")
        classifier_label.grid(row=0, column=0, sticky="nw")

        classifier_combobox = ttk.Combobox(self, textvariable=self.classifier_var, values=["Perceptron", "Logistic Regression", "SVM", "Decision Tree", "Random Forest", "Boosting", "Multilayer Perceptron"])
        classifier_combobox.grid(row=1, column=0, sticky="nw")

        # Step selection
        steps_label = ttk.Label(self, text="Select Step:")
        steps_label.grid(row=2, column=0, sticky="nw")

        steps_combobox = ttk.Combobox(self, textvariable=self.steps_var, values=["Train and Analyze", "Grid Search"])
        steps_combobox.grid(row=3, column=0, sticky="nw")

        # Execute button
        execute_button = ttk.Button(self, text="Execute", command=self.execute)
        execute_button.grid(row=4, column=0, sticky="nw")

        # Classification report and result area
        self.result_text = tk.Text(self, height=40, width=50,font=("Helvetica", 10))
        self.result_text.grid(row=5, column=0, sticky="nw")

        self.show_classification(None)
        
    def execute(self):
        classifier = self.classifier_var.get()
        steps = self.steps_var.get()
        
        if steps == "Train and Analyze":
            # Load the data
            X, y = load_data()

            # # Define the classifier
            # if classifier == "Perceptron":
            #     classifier = Perceptron()
            # elif classifier == "Logistic Regression":
            #     classifier = LogisticRegression()
            # elif classifier == "SVM":
            #     classifier = SVC()
            # elif classifier == "Decision Tree":
            #     classifier = DecisionTreeClassifier()
            # elif classifier == "Random Forest":
            #     classifier = RandomForestClassifier()
            # elif classifier == "Boosting":
            #     classifier = GradientBoostingClassifier(n_estimators=10,n_iter_no_change=10)
            # elif classifier == "Multilayer Perceptron":
            #     classifier = MLPClassifier()

            # Train the model and perform analysis
            classification_report, cm = train(X, y, classifier)

            # Display the classification report and result in the text area
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Classification Report:\n")
            self.result_text.insert(tk.END, classification_report)
            self.result_text.insert(tk.END, "\nConfusion Matrix:\n")
            self.show_classification(cm)
            
        elif steps == "Grid Search":
            # Load the data
            X, y = load_data()

            # Define the classifier
            # if classifier == "Perceptron":
            #     classifier = Perceptron()
            # elif classifier == "Logistic Regression":
            #     classifier = LogisticRegression()
            # elif classifier == "SVM":
            #     classifier = SVC()
            # elif classifier == "Decision Tree":
            #     classifier = DecisionTreeClassifier()
            # elif classifier == "Random Forest":
            #     classifier = RandomForestClassifier()
            # elif classifier == "Boosting":
            #     classifier = GradientBoostingClassifier()
            # elif classifier == "Multilayer Perceptron":
            #     classifier = MLPClassifier()

            # Perform grid search
            best_params, best_score = grid_searching(X, y, classifier)
            
            # Display the best hyperparameters and score in the text area
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Best Hyperparameters and Preprocessing Options:\n")
            self.result_text.insert(tk.END, str(best_params))
            self.result_text.insert(tk.END, "\nBest Score:\n")
            self.result_text.insert(tk.END, str(best_score))
            
  

    def show_classification(self, cm):
        if cm is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
        else:
            plt.figure(figsize=(6, 4))
            normalized_cm = cm / cm.sum(axis=1, keepdims=True)
            sns.heatmap(normalized_cm, annot=True, fmt='.2f', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')

        # Create a Tkinter canvas to display the plot
        self.canvas = tkagg.FigureCanvasTkAgg(plt.gcf(), master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, column=1, rowspan=4, sticky="nsew", padx=10, pady=10)

        # # Create a Tkinter toolbar for the plot
        # self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas, self)
        # self.toolbar.update()
        # self.canvas.get_tk_widget().grid(row=6, column=0, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.mainloop()
