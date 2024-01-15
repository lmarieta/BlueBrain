import csv
import glob
import os
import re
from pymysql.converters import escape_string

# Add JSON files of acells to the database
cell_strings = {}

# Change path depending on where the JSON files you want to add are
json_files_path = '/home/lucas/BBP/Data/jsonData/*.json'
for file_path in glob.glob(json_files_path):
    # Extract the cell ID or some identifier from the file name or path
    file_name = os.path.basename(file_path).replace('.json', '')
    cell_id_match = re.match(r'aCell(\d+_?\d*)', file_name)
    if cell_id_match:
        cell_id = cell_id_match.group(1)

        # Read the JSON file and store its content as a string
        with open(file_path, 'r') as json_file:
            json_data = json_file.read()
            json_data = escape_string(json_data)
            cell_strings[cell_id] = json_data

# Load file to find equivalence between acell CellID and cell.name
cell_list_file_name = '/home/lucas/BBP/Code/CellList30-May-2022.csv'
sql_output_file_name = '/home/lucas/BBP/Code/update_db.sql'

with open(cell_list_file_name, 'r') as csv_file, open(sql_output_file_name, 'r') as sql_output_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        # Access columns by their header names
        cell_id = row['CellID']
        cell_name = row[' Path']

        if cell_id in cell_strings:
            json_string = cell_strings[cell_id]

            # Regular expression pattern
            pattern = r'\\([^\\ ]*?)(?:\\| )'

            # Find matches using re.findall
            matches = re.findall(pattern, cell_name)

            # Generate SQL statements
            sql_output_file.write(f"UPDATE cells SET acell_json = '{json_string}' WHERE name = '{matches[0]}';\n")

print(f"SQL script generated at: {sql_output_file_name}")
