def discretizer(var):
    if var >= 80:
        return "high"
    if var >= 50 and var <= 79:
        return "medium"
    if var >= 0 and var <= 49:
        return "low"

def get_column(table, header, col_name):
    col = []
    col_index = header.index(col_name)
    for row in table:
        col.append(row[col_index])

    return col

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
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