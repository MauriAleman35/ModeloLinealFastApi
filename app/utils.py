import calendar
from datetime import datetime, timedelta
import numpy as np

def get_month_name(month_number):
    """Convierte número de mes a nombre"""
    return calendar.month_name[month_number]

def get_future_dates(months=3, start_date=None):
    """Genera fechas futuras para predicción"""
    if not start_date:
        start_date = datetime.now()
    
    dates = []
    current_date = start_date
    
    for i in range(months):
        # Avanzar al siguiente mes
        if current_date.month == 12:
            next_month = 1
            next_year = current_date.year + 1
        else:
            next_month = current_date.month + 1
            next_year = current_date.year
        
        current_date = datetime(next_year, next_month, 1)
        
        dates.append({
            'year': current_date.year,
            'month': current_date.month,
            'month_name': calendar.month_name[current_date.month]
        })
    
    return dates