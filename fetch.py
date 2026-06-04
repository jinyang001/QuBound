import json
from datetime import datetime, timedelta
from typing import List
from qiskit_ibm_provider import IBMProvider


provider = IBMProvider()
# backend = provider.get_backend('ibm_auckland')
# backend = provider.get_backend('ibm_hanoi')
# backend = provider.get_backend('ibm_cairo')
# backend = provider.get_backend('ibm_algiers')
# backend = provider.get_backend('ibmq_mumbai')
# backend = provider.get_backend('ibmq_kolkata')
# backend = provider.get_backend('ibmq_guadalupe')
backend = provider.get_backend('ibm_nairobi')
def datetime_converter(o):
    """Helper function to convert datetime objects to strings."""
    if isinstance(o, datetime):
        return o.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def store_backend_properties(start_date: datetime, num_days: int, hours_list: List[int], file_name: str):
    """
    Fetch backend properties for given dates and hours, and store them in a JSON file.
    """
    properties_dict = {}
    current_date = start_date

    for _ in range(num_days):
        for hour in hours_list:
            t = current_date.replace(hour=hour)
            timestamp_str = t.strftime('%Y-%m-%d %H:%M:%S')
            try:
                properties = backend.properties(datetime=t)
                # Convert properties to dictionary
                properties_as_dict = properties.to_dict()
                properties_dict[timestamp_str] = properties_as_dict
            except Exception as e:
                print(f"Error fetching properties for {timestamp_str}: {e}")

        current_date -= timedelta(days=1)

    # Save to a file
    with open(file_name, 'w') as f:
        json.dump(properties_dict, f, default=datetime_converter)  # use the datetime_converter

    print(f"Stored backend properties in {file_name}")



start_date = datetime(day=15, month=10, year=2023)
store_backend_properties(start_date, 10, [0, 6, 12, 18], "properties/test.json")
