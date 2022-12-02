def discretizer(var):
    if var >= 80:
        return "high"
    if var >= 50 and var <= 79:
        return "medium"
    if var >= 0 and var <= 49:
        return "low"