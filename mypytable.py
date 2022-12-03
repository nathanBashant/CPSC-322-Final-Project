"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains the MyPyTable class, which is used for data storage.
"""
import copy
import csv
import statistics

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(self.get_pretty_print())


    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        col_data = []
        for row in self.data:
            if include_missing_values or (len(str(row[col_index])) != 0 and \
                row[col_index] != "NA" and row[col_index] != "N/A"):
                col_data.append(row[col_index])
        return col_data

    def get_other_columns(self, exclude_col_names):
        """Returns a MyPyTable with all columns except for the column(s) specified.

        Args:
            exclude_col_names(list of str): names of columns to be excluded.

        Returns:
            MyPyTable: table with all columns of data except the ones to be excluded.
        """
        data = []
        for row in self.data:
            row_data = []
            for col_index, column in enumerate(row):
                if self.column_names[col_index] not in exclude_col_names:
                    row_data.append(column)
            data.append(row_data)
        col_names = []
        for col_name in self.column_names:
            if col_name not in exclude_col_names:
                col_names.append(col_name)
        return MyPyTable(col_names, data)


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row_index, row_data in enumerate(self.data):
            for val_index, value in enumerate(row_data):
                try:
                    numeric_value = float(value)
                    self.data[row_index][val_index] = numeric_value
                except ValueError:
                    pass

    def convert_to_str(self, col_name):
        """Convert each value in a given column to str.

        Args:
            col_name(str) = name of column to convert the values of.
        """
        col_index = self.column_names.index(col_name)
        for row_index, row_data in enumerate(self.data):
            # Removes the decimal if the value is a float.
            try:
                numeric_value = int(row_data[col_index])
                self.data[row_index][col_index] = numeric_value
            except ValueError:
                pass
            self.data[row_index][col_index] = str(row_data[col_index])

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for i in range(len(row_indexes_to_drop) - 1, -1, -1):
            self.data.pop(row_indexes_to_drop[i])

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        temp_data = []
        with open(filename, "r") as infile:
            reader = csv.reader(infile)
            for row in reader:
                temp_data.append(row)
        col_names = temp_data.pop(0)
        self.data = temp_data
        self.column_names = col_names
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        col_list = [self.get_column(col_name) for col_name in key_column_names]
        unique_rows = []
        indexes_of_duplicates = []
        for i in range(len(col_list[0])):
            temp_list = []
            for col_data in col_list:
                temp_list.append(col_data[i])
            if temp_list in unique_rows:
                indexes_of_duplicates.append(i)
            else:
                unique_rows.append(temp_list)
        return indexes_of_duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA" or "N/A" or "").
        """
        for i in range(len(self.data) - 1, -1, -1):
            for value in self.data[i]:
                if len(str(value)) == 0 or value == "NA" or value == "N/A":
                    self.data.pop(i)
                    break

    def remove_rows_with_missing_values_in_column(self, col_name):
        """Remove rows from the table data that contain a missing value ("NA" or "N/A" or "")
            in a given column.

            Args:
                col_name(str): name of column to check for missing values
        """
        col_index = self.column_names.index(col_name)
        for i in range(len(self.data) - 1, -1, -1):
            value = self.data[i][col_index]
            if len(str(value)) == 0 or value == "NA" or value == "N/A":
                self.data.pop(i)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col_sum = 0
        missing_val_index_list = []
        for index, row in enumerate(self.data):
            if row[col_index] == "NA":
                missing_val_index_list.append(index)
            else:
                col_sum += row[col_index]
        col_avg = col_sum / (len(self.data) - len(missing_val_index_list))
        for missing_val_index in missing_val_index_list:
            self.data[missing_val_index][col_index] = col_avg

    def replace_missing_vals_in_columns(self, col_names):
        """For columns with continuous data, fill missing values in the given columns
        by the each column's original average. Makes calls to
        replace_missing_values_with_column_average() to do so.

        Args:
            col_names(str): list of names of columns to fill with the original
            average (of the column).

        Notes:
            Just a helper function for replacing missing values in multiple columns.
        """
        for col_name in col_names:
            self.replace_missing_values_with_column_average(col_name)

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_stats_col_names = ["attribute", \
            "min", "max", "mid", "avg", "median"]
        summary_stats_data = []
        for col_name in col_names:
            col_data = self.get_column(col_name, False)
            if len(col_data) != 0:
                col_stats = [col_name]
                col_stats.append(min(col_data))
                col_stats.append(max(col_data))
                col_stats.append((max(col_data) + min(col_data)) / 2)
                col_stats.append(sum(col_data) / len(col_data))
                col_data.sort()
                col_stats.append(statistics.median(col_data))
                summary_stats_data.append(col_stats)
        return MyPyTable(summary_stats_col_names, summary_stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_data = []
        col_indexes = [self.column_names.index(col_name) for \
            col_name in key_column_names]
        other_col_indexes = [other_table.column_names.index(col_name) for \
            col_name in key_column_names]
        for row in self.data:
            for other_row in other_table.data:
                matching = True
                for i in range(len(key_column_names)):
                    if row[col_indexes[i]] != other_row[other_col_indexes[i]]:
                        matching = False
                if matching:
                    joined_row = []
                    joined_row.extend(row)
                    for i in range(len(other_table.column_names)):
                        if i not in other_col_indexes:
                            joined_row.append(other_row[i])
                    joined_data.append(joined_row)
        joined_column_names = []
        for col_name in self.column_names:
            joined_column_names.append(col_name)
        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                joined_column_names.append(col_name)
        return MyPyTable(joined_column_names, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # 2D list to save on local variable count
        # header in first list, data in second
        joined_table = [[], []]
        # used rows from self in first list, used rows from other table in second
        used_row_indexes = [[],[]]
        col_indexes = [self.column_names.index(col_name) for \
            col_name in key_column_names]
        other_col_indexes = [other_table.column_names.index(col_name) for \
            col_name in key_column_names]
        for index, row in enumerate(self.data):
            for other_index, other_row in enumerate(other_table.data):
                matching = True
                for i in range(len(key_column_names)):
                    if row[col_indexes[i]] != other_row[other_col_indexes[i]]:
                        matching = False
                if matching:
                    joined_row = []
                    joined_row.extend(row)
                    for i in range(len(other_table.column_names)):
                        if i not in other_col_indexes:
                            joined_row.append(other_row[i])
                    joined_table[1].append(joined_row)
                    used_row_indexes[0].append(index)
                    used_row_indexes[1].append(other_index)
        # creating the joined table's list of column names
        for col_name in self.column_names:
            joined_table[0].append(col_name)
        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                joined_table[0].append(col_name)
        # adding non-matching elements
        joined_table[1] = self.find_non_matching_rows(used_row_indexes[0],
            joined_table[0], "NA", joined_rows=joined_table[1])
        joined_table[1] = other_table.find_non_matching_rows(used_row_indexes[1],
            joined_table[0], "NA", joined_rows=joined_table[1])
        return MyPyTable(joined_table[0], joined_table[1])

    def find_non_matching_rows(self, used_indexes, joined_col_names,
        missing_val, joined_rows):
        """Return a list of rows that did not match any rows in the other
        table used in the join.

        Args:
            table(MyPyTable): the table to get rows from.
            used_indexes(list of int): list of used indexes.
            joined_col_names(list of str): list of column names
            from the joined table.

        Returns:
            list: list of the joined data with the non-matching rows added.

        Notes:
            To be called during a full outer join.
            Pad the attributes with missing values with missing_val.
        """
        for index, row in enumerate(self.data):
            if index not in used_indexes:
                temp_row = []
                for col_name in joined_col_names:
                    if col_name in self.column_names:
                        temp_row.append(row[self.column_names.index(col_name)])
                    else:
                        temp_row.append(missing_val)
                joined_rows.append(temp_row)
        return joined_rows

    def get_column_frequencies(self, col_name):
        """Calculates the frequencies for a given column in the dataset.

        Args:
            col_name(str): name of the column to calculate the frequencies of.

        Returns:
            values: list of unique values from the column.
            counts: list of the number of occurences for each value.
        """
        col_data = self.get_column(col_name, False)
        col_data.sort()
        values = []
        counts = []
        for value in col_data:
            if value not in values:
                values.append(value)
                counts.append(1)
            else:
                counts[-1] += 1
        return values, counts
