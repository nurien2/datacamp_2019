import os
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class FeatureExtractor(object):
    def __init__(self):
        
        path = os.path.dirname(__file__)
        path_health = os.path.join(path, 'health.csv')
        self.health = pd.read_csv(path_health, low_memory=False)

        path_population = os.path.join(path, 'population.csv')
        self.population = pd.read_csv(path_population, low_memory=False)
        
        self.population = self.population[['Country', 'Sex', 'Age', 'Time', 'Value']]
        self.population = self.population.rename(columns={"Time": "Year"})
        self.population = self.population[self.population["Sex"].isin(["Men", "Women"]) & (self.population["Age"] == "Total")]
        self.population.loc[:,"Sex"] = self.population["Sex"].map({"Men": 0, "Women": 1})
        self.population = self.population.groupby(["Country", 'Year']).agg({'Value' : "sum"}).reset_index()

        self.health = self.health[['Financing scheme', 'Function', 'Measure', 'Country', 'Year', 'Value']]
        self.health = self.health[(self.health["Function"] == "Current expenditure on health (all functions)") &
                    (self.health["Financing scheme"] == "All financing schemes") &
                    (self.health["Measure"] == "Share of gross domestic product")]
        self.health = self.health.groupby(["Country", "Year"]).agg({"Value": "sum"}).reset_index()

        
        self.categorical = ['Country of birth/nationality', 'Variable', 'Gender', 'Country']
        self.numeric = ['Year', "Value_Health", "Value_Population"]
        
        self.target_encoder = ce.target_encoder.TargetEncoder()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('category_variables', make_pipeline(self.target_encoder, SimpleImputer(strategy='median')), self.categorical),
                ('numeric_variables', SimpleImputer(strategy='median'), self.numeric)])
        
        
    def merge(self, X_df):
        
        X_df = X_df.merge(self.health[["Country", 'Year', 'Value']], how='left', on=["Country", "Year"])   
        X_df = X_df.merge(self.population[["Country", 'Year', 'Value']], how='left', on=["Country", "Year"], sort=False, suffixes=("_Health", '_Population'))
        return(X_df)
        
    
    def fit(self, X_df, y_array):
        
        X_df = self.merge(X_df)
        self.preprocessor.fit(X_df, y_array)

    def transform(self, X_df):
        
        X_df = self.merge(X_df)
        X_array = self.preprocessor.transform(X_df)
        return X_array
    
    def fit_transform(self, X_df, y_array):
        self.fit(X_df, y_array)
        return(self.transform(X_df))