import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from ml_code import load_data, train_model, save_model, load_saved_model, predict_new_data
import joblib

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Fraud Detection System")
        self.root.geometry("800x600")
        
        self.model = None
        self.data = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Notebook for multiple tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Data Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data")
        self.setup_data_tab()
        
        # Model Tab
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model")
        self.setup_model_tab()
        
        # Prediction Tab
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Prediction")
        self.setup_prediction_tab()
    
    def setup_data_tab(self):
        # Load Data Section
        ttk.Label(self.data_tab, text="Load Dataset").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        self.data_path = tk.StringVar()
        ttk.Entry(self.data_tab, textvariable=self.data_path, width=50).grid(row=1, column=0, padx=10, pady=5)
        
        ttk.Button(self.data_tab, text="Browse", command=self.browse_data).grid(row=1, column=1, padx=10, pady=5)
        ttk.Button(self.data_tab, text="Load Data", command=self.load_data).grid(row=2, column=0, padx=10, pady=5)
        
        # Data Info Section
        ttk.Label(self.data_tab, text="Data Information").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        
        self.data_info = tk.Text(self.data_tab, height=15, width=80)
        self.data_info.grid(row=4, column=0, columnspan=2, padx=10, pady=5)
        
        # Scrollbar for data info
        scrollbar = ttk.Scrollbar(self.data_tab, command=self.data_info.yview)
        scrollbar.grid(row=4, column=2, sticky='ns')
        self.data_info['yscrollcommand'] = scrollbar.set
    
    def setup_model_tab(self):
        # Train Model Section
        ttk.Label(self.model_tab, text="Train Model").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        ttk.Button(self.model_tab, text="Train New Model", command=self.train_model).grid(row=1, column=0, padx=10, pady=5)
        
        # Save/Load Model Section
        ttk.Label(self.model_tab, text="Model Operations").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        
        self.model_path = tk.StringVar(value="model/fraud_model.pkl")
        ttk.Entry(self.model_tab, textvariable=self.model_path, width=50).grid(row=3, column=0, padx=10, pady=5)
        
        ttk.Button(self.model_tab, text="Browse", command=self.browse_model).grid(row=3, column=1, padx=10, pady=5)
        ttk.Button(self.model_tab, text="Save Model", command=self.save_model).grid(row=4, column=0, padx=10, pady=5)
        ttk.Button(self.model_tab, text="Load Model", command=self.load_model).grid(row=4, column=1, padx=10, pady=5)
        
        # Model Info Section
        ttk.Label(self.model_tab, text="Model Information").grid(row=5, column=0, padx=10, pady=5, sticky='w')
        
        self.model_info = tk.Text(self.model_tab, height=10, width=80)
        self.model_info.grid(row=6, column=0, columnspan=2, padx=10, pady=5)
        
        # Scrollbar for model info
        scrollbar = ttk.Scrollbar(self.model_tab, command=self.model_info.yview)
        scrollbar.grid(row=6, column=2, sticky='ns')
        self.model_info['yscrollcommand'] = scrollbar.set
    
    def setup_prediction_tab(self):
        # Manual Input Section
        ttk.Label(self.prediction_tab, text="Enter Transaction Details").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        # Create input fields for each feature (simplified example)
        self.input_fields = {}
        features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                    'V28', 'Amount']
        
        for i, feature in enumerate(features):
            ttk.Label(self.prediction_tab, text=feature).grid(row=i+1, column=0, padx=10, pady=2, sticky='e')
            self.input_fields[feature] = ttk.Entry(self.prediction_tab)
            self.input_fields[feature].grid(row=i+1, column=1, padx=10, pady=2)
        
        ttk.Button(self.prediction_tab, text="Predict", command=self.predict_manual).grid(row=len(features)+1, column=0, columnspan=2, pady=10)
        
        # Prediction Result Section
        ttk.Label(self.prediction_tab, text="Prediction Result").grid(row=len(features)+2, column=0, padx=10, pady=5, sticky='w')
        
        self.prediction_result = tk.Text(self.prediction_tab, height=5, width=60)
        self.prediction_result.grid(row=len(features)+3, column=0, columnspan=2, padx=10, pady=5)
        
        # File Prediction Section
        ttk.Label(self.prediction_tab, text="Or Predict from File").grid(row=len(features)+4, column=0, padx=10, pady=5, sticky='w')
        
        self.predict_file_path = tk.StringVar()
        ttk.Entry(self.prediction_tab, textvariable=self.predict_file_path, width=50).grid(row=len(features)+5, column=0, padx=10, pady=5)
        
        ttk.Button(self.prediction_tab, text="Browse", command=self.browse_predict_file).grid(row=len(features)+5, column=1, padx=10, pady=5)
        ttk.Button(self.prediction_tab, text="Predict from File", command=self.predict_from_file).grid(row=len(features)+6, column=0, columnspan=2, pady=10)
    
    # Data Tab Functions
    def browse_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.data_path.set(filepath)
    
    def load_data(self):
        try:
            self.data = load_data(self.data_path.get())
            info = f"Data loaded successfully!\n\n"
            info += f"Number of transactions: {len(self.data)}\n"
            info += f"Number of features: {len(self.data.columns)}\n"
            info += f"Fraudulent transactions: {sum(self.data['Class'])} ({sum(self.data['Class'])/len(self.data)*100:.2f}%)\n\n"
            info += f"First 5 rows:\n{self.data.head().to_string()}"
            
            self.data_info.delete(1.0, tk.END)
            self.data_info.insert(tk.END, info)
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    # Model Tab Functions
    def browse_model(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if filepath:
            self.model_path.set(filepath)
    
    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.model = train_model(self.data)
            info = "Model trained successfully!\n\n"
            info += f"Model type: {type(self.model).__name__}\n"
            
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, info)
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save! Train or load a model first.")
            return
        
        try:
            save_model(self.model, self.model_path.get())
            messagebox.showinfo("Success", f"Model saved to {self.model_path.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            self.model = load_saved_model(self.model_path.get())
            info = "Model loaded successfully!\n\n"
            info += f"Model type: {type(self.model).__name__}\n"
            
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, info)
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    # Prediction Tab Functions
    def browse_predict_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.predict_file_path.set(filepath)
    
    def predict_manual(self):
        if self.model is None:
            messagebox.showerror("Error", "No model loaded! Please train or load a model first.")
            return
        
        try:
            # Collect input values
            input_data = {}
            for feature, entry in self.input_fields.items():
                value = entry.get()
                if not value:
                    value = 0.0  # Default value if empty
                input_data[feature] = float(value)
            
            # Convert to DataFrame (assuming your model expects this format)
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = predict_new_data(self.model, input_df)
            
            # Display result
            result = "Normal Transaction" if prediction[0] == 0 else "FRAUDULENT TRANSACTION!"
            color = "green" if prediction[0] == 0 else "red"
            
            self.prediction_result.delete(1.0, tk.END)
            self.prediction_result.insert(tk.END, result)
            self.prediction_result.tag_configure("color", foreground=color)
            self.prediction_result.tag_add("color", "1.0", "end")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def predict_from_file(self):
        if self.model is None:
            messagebox.showerror("Error", "No model loaded! Please train or load a model first.")
            return
        
        try:
            # Load data to predict
            data = pd.read_csv(self.predict_file_path.get())
            
            # Make predictions
            predictions = predict_new_data(self.model, data)
            
            # Add predictions to data
            data['Prediction'] = predictions
            
            # Save results
            output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_path:
                data.to_csv(output_path, index=False)
                messagebox.showinfo("Success", f"Predictions saved to {output_path}")
                
                # Show summary
                fraud_count = sum(predictions)
                total = len(predictions)
                summary = f"Prediction Summary:\n"
                summary += f"Total transactions: {total}\n"
                summary += f"Fraudulent transactions: {fraud_count} ({fraud_count/total*100:.2f}%)\n"
                
                self.prediction_result.delete(1.0, tk.END)
                self.prediction_result.insert(tk.END, summary)
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()