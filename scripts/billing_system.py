def calculate_bill(consumption_kwh):
    """
    Tier-based pricing:
    0 - 100 kWh: $0.12 / kWh
    101 - 300 kWh: $0.15 / kWh
    301 - 500 kWh: $0.20 / kWh
    Above 500 kWh: $0.25 / kWh
    """
    bill = 0
    if consumption_kwh <= 100:
        bill = consumption_kwh * 0.12
    elif consumption_kwh <= 300:
        bill = (100 * 0.12) + (consumption_kwh - 100) * 0.15
    elif consumption_kwh <= 500:
        bill = (100 * 0.12) + (200 * 0.15) + (consumption_kwh - 300) * 0.20
    else:
        bill = (100 * 0.12) + (200 * 0.15) + (200 * 0.20) + (consumption_kwh - 500) * 0.25
        
    return round(bill, 2)

if __name__ == "__main__":
    # Test cases
    print(f"Bill for 50 kWh: ${calculate_bill(50)}")
    print(f"Bill for 250 kWh: ${calculate_bill(250)}")
    print(f"Bill for 450 kWh: ${calculate_bill(450)}")
    print(f"Bill for 600 kWh: ${calculate_bill(600)}")
