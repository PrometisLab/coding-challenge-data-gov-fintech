import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer():
    
    def __init__(self,filepath):
        self.filepath=filepath
        self.dataframe=pd.read_csv(filepath,index_col=0,delimiter=";")
        
    def validate_ptir(self):
        income=np.array(self.dataframe["income"])/12
        payment=np.array(self.dataframe["monthly_payment"])
        ptir=np.array(self.dataframe["payment_to_income_ratio"])
        ptir_custom=np.round(payment/income,3)
        print(f"ptir={ptir}")
        print(f"ptir_custom={ptir_custom}")
        print(np.mean(ptir==ptir_custom))
        
        
        
    def ptir_vs_ls(self):
        plt.figure()
        plt.plot(self.dataframe["payment_to_income_ratio"],self.dataframe["payment_to_income_ratio"])

    