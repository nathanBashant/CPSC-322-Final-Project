def discretizer(var):
    if var >= 80:
        return "excellent"
    if var >= 50:
        return "fair"
    return "low"

def get_frequencies_col(col_name):
    col = col_name
    col.sort() 
    
    values = [] 
    counts = []
    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts