import pandas as pd
import numpy as np
import os

def generate_dataset(num_samples=1000):
    np.random.seed(42)
    
    # Features
    household_size = np.random.randint(1, 8, num_samples)
    num_appliances = np.random.randint(5, 25, num_samples)
    daily_usage_hours = np.random.uniform(2, 18, num_samples)
    temperature = np.random.uniform(15, 40, num_samples)
    prev_month_consumption = np.random.uniform(100, 800, num_samples)
    
    # Seasonal factor (1: Winter, 2: Spring, 3: Summer, 4: Autumn)
    seasonal_factor = np.random.randint(1, 5, num_samples)
    
    # Base rate per kWh
    unit_rate_per_kwh = np.full(num_samples, 0.15) 

    # Target variable: Actual Electricity Consumption (kWh)
    # A simple formula to make it somewhat realistic but noisy
    # Consumption = (H_size * 20) + (App * 10) + (Usage * 30) + (Temp * 2) + offset
    consumption = (household_size * 25) + (num_appliances * 12) + \
                  (daily_usage_hours * 35) + (temperature * 1.5) + \
                  (seasonal_factor * 20) + np.random.normal(0, 20, num_samples)
    
    # Ensure consumption is positive
    consumption = np.maximum(consumption, 50)

    df = pd.DataFrame({
        'household_size': household_size,
        'num_appliances': num_appliances,
        'daily_usage_hours': daily_usage_hours,
        'temperature': temperature,
        'prev_month_consumption': prev_month_consumption,
        'seasonal_factor': seasonal_factor,
        'unit_rate_per_kwh': unit_rate_per_kwh,
        'actual_consumption': consumption
    })
    
    output_path = r'C:\Users\velus\.gemini\antigravity\scratch\electricity_prediction\data\electricity_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}")

if __name__ == "__main__":
    generate_dataset()
