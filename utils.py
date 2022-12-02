def discretizer(var):
    if var >= 80 & var <= 100:
        return "high"
    elif var >= 50 & var <= 80:
        return "medium"
    elif var >= 0 & var <= 49:
        return "low"