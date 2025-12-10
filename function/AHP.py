import json
import numpy as np
import pandas as pd 
from pathlib import Path


class GetEigenValues:
    def __init__(self, variable, init_score):
        super().__init__()
        self.variable = variable
        self.init_score = init_score
        self.ROOT_DIR = Path().resolve()
    
    
    def run_calculation(self) -> dict:
        try:
            self.load_RI()
            self.score_final_cal()
            self.score_norm_cal()
            self.score_eigen_cal()
            
            return self.score_eigen_cal()
        
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            
            return {
                "Eigenvalue Maximum": 0,
                "CI (Consistency Index)": 0,
                "CR (Consistency Ratio)": 0
            }
            
    
    def load_RI(self):
        def convert_keys_to_int(obj):
            if isinstance(obj, dict):
                return {int(k) if k.isdigit() else k: v for k, v in obj.items()}
            
            return obj

        path_RI_file = f'{self.ROOT_DIR}/data_static/Ratio_Inconsistency.json'

        with open(path_RI_file, 'r') as f:
            RI = json.load(f, object_hook=convert_keys_to_int)
        
        self.RI = RI
        
        return RI
    
    
    def score_final_cal(self):
        df_score = {}

        for val1 in self.variable:
            for val2 in self.variable:
                if val1 != val2:
                    if (val1, val2) in list(self.init_score.keys()):
                        df_score[(val1, val2)] = self.init_score[(val1, val2)]
                    else:
                        df_score[(val1, val2)] = 1 / self.init_score[(val2, val1)]
                else:
                    df_score[(val1, val1)] = 1


        # Convert the dictionary into a Series and Convert to a DataFrame matrix form
        df_score_final = pd.Series(df_score).unstack()
        df_score_final.loc['Total'] = df_score_final.sum(axis=0)
        
        self.df_score_final = df_score_final
        
        return df_score_final
    
    
    def score_norm_cal(self):
        df_score_norm = self.df_score_final.div(self.df_score_final.loc['Total'], axis=1)
        df_score_norm['Total'] = df_score_norm.sum(axis=1) / len(self.variable)
        
        self.df_score_norm = df_score_norm
        
        return df_score_norm
    
    
    def score_eigen_cal(self) -> dict:
        df_score_calc = self.df_score_final.copy().iloc[0:len(self.variable), :].T
        df_score_calc['Total'] = self.df_score_norm['Total']
        
        df_eigen = {}
        eigen_value = 0

        for val in self.variable:
            df_eigen[val] = float(np.sum(df_score_calc[val] * df_score_calc['Total'])) / float(df_score_calc['Total'].loc[val])
            eigen_value += float(np.sum(df_score_calc[val] * df_score_calc['Total']))

        df_eigen["Eigenvalue Maximum"] = eigen_value
        df_eigen["CI (Consistency Index)"] = (eigen_value - len(self.variable)) / (len(self.variable) - 1)
        df_eigen["CR (Consistency Ratio)"] = df_eigen["CI (Consistency Index)"] / self.RI[len(self.variable)]
        df_eigen_final = pd.Series(df_eigen)
        
        self.df_eigen = df_eigen
        self.df_eigen_final = df_eigen_final
        
        return df_eigen
            