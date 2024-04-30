import re
import pandas as pd
import xlsxwriter
from datetime import datetime

from scripts import word_to_num, add_commas, compose_packaging_pattern, extract_quantities, extract_weight_in_kg, \
    is_singular, is_hazardous, check_for_relevance, remove_singular_quantities, remove_plural_weights, \
    add_missing_weights

from knowlege_base import cargo_items, packaging_plural

from inputs import inputs


def remove_duplicates(entities):
    """
    Removes duplicate entities from a dictionary based on their values. This function iterates through the provided
    dictionary, identifying entries that have identical values. If multiple keys have the same value, all but one
    instance of these duplicates are removed, ensuring that each value in the dictionary is unique.

    Args:
        entities (dict): A dictionary where the keys are labels (such as entity names) and the values are the
        corresponding data points that need deduplication.

    Returns:
        dict: An updated dictionary where all duplicate values are removed, leaving only unique values.
        Each unique value is retained under one of its original keys.
    """

    for key, value in entities.items():
        if isinstance(value, list):
            seen = set()
            new_list = []
            for item in value:
                if item not in seen:
                    new_list.append(item)
                    seen.add(item)
            entities[key] = new_list

    return entities

def extract_entities(input_text):
    """
    Extracts various entities related to cargo from a provided input_text, utilizing a series of specialized functions.
    This function processes the input text to identify and extract multiple types of cargo-related information,
    including cargo items, quantities, packaging types, and weights.


    Key steps include:
        1. **Normalization**: Converts numerical words to numeric values to standardize data entries.
        2. **Comma Insertion**: Adds commas after numbers and packaging types to clarify the separation of listed
        quantities.
        3. **Entity Identification**: Uses regular expressions to match cargo types, packaging, and weights from the
        structured text.
        4. **Quantitative Extraction**: Gathers numerical data about cargo quantities and weights, converting and
        categorizing them by packaging type.
        5. **Hazard Assessment**: Evaluates the text to determine if the cargo has hazardous properties based on
        specific keywords and context.
        6. **Duplication Removal**: Applies a function to remove duplicate entries.

    The final output is a dictionary containing clean, structured data about cargo types, packaging, weights, and
    hazardous properties, with duplicates removed and missing information flagged.

    Args:
        input_text (str): A string containing a description of cargo, including details about weight and packaging.

    Returns:
        dict: An updated dictionary where all duplicate values are removed, each unique value is retained under one of
        its original keys, and missing information is addressed.
    """
    # Convert number words to numeric values
    input_text = ' '.join(
        [str(word_to_num(word)) if word_to_num(word) is not None else word for word in input_text.split()])

    # Add comas to a text to separate quantities
    input_text = add_commas(input_text)

    # Extract cargo_matches using patterns
    cargo_pattern = r"(" + "|".join(cargo_items) + ")"
    cargo_matches = re.findall(cargo_pattern, input_text)

    # Extract quantity_matches using patterns
    packaging_pattern, _ = compose_packaging_pattern(packaging_plural)
    packaging_matches = re.findall(packaging_pattern, input_text)
    quantity_matches = extract_quantities(input_text)

    # Extract weight_matches using patterns
    weight_matches = extract_weight_in_kg(input_text)

    # Create separate categories for each type of packaging
    packaging_contents = {}
    packaging_quantities = {}
    for packaging in packaging_matches:
        packaging_quantities[f"quantity_{packaging}"] = 0
        packaging_contents[f"{packaging}_contain"] = 0

    weight_units_pattern = r'(?:kg|kilograms|tonnes|pounds|tonne|kilogram|pound|grams|gram)'
    for quantity, packaging in zip(quantity_matches, packaging_matches):

        # Iterate over matches in the input text, to find additional context
        for match in re.finditer(r'\b(?:with|containing|holding|contains|capacity|hold)\b\s+(\d+)\s+(?:'
                                 + weight_units_pattern + r'\b)?\s*(' + '|'.join(packaging_matches) + r')?',
                                 input_text):

            if re.search(weight_units_pattern, match.group(0)):
                #print("weight_units_pattern found... aborting")
                continue

            elif re.search(r'\b(\d+)\b', match.group(0)):
                print("weight_units_pattern not found... continue")
                phrase = match.group(0)

                quantity_contents = re.search(r'\b(\d+)\b', phrase).group(0)
                match = re.search(rf'\b{re.escape(quantity_contents)}\s+(\w+)', input_text)
                if match:
                    next_word = match.group(1)
                for packaging in packaging_matches:
                    if is_singular(packaging):
                        continue
                    if f"quantity_{packaging}" in packaging_quantities:
                        packaging_contents[f"{packaging}_contain"] = f"{quantity_contents} {next_word}"

        if f"quantity_{packaging}" in packaging_quantities:
            quantity_value, packaging = quantity
            packaging_quantities[f"quantity_{packaging}"] = str(quantity_value)

    for key, value in packaging_quantities.items():

        if value == str(0):
            packaging_quantities[key] = "information missing"

    # Check for misidentified packaging_contents keys
    keys_to_delete = [key for key, value in packaging_contents.items() if
                      isinstance(value, (int, float)) and value == 0 and not isinstance(value, bool)]
    for key in keys_to_delete:
        del packaging_contents[key]

    # Check if sentences imply that cargo is hazardous
    sentences = re.split(r'(?<=[.!?]) +', input_text)
    hazardous = False
    for sentence in sentences:

        if is_hazardous(sentence):
            hazardous = True
        elif hazardous:
            continue
        else:
            hazardous = False

    # Combine entities into dictionary
    entities = {
        "Cargo": cargo_matches,
        "Packaging": packaging_matches,
        **packaging_quantities,
        **packaging_contents,
        "Weight": weight_matches,
        "Hazardous properties": hazardous
    }
    # Check if any entities have value None or empty assigned
    for key, value in entities.items():
        if value is None or (isinstance(value, list) and not value):
            entities[key] = "information missing"

    return remove_duplicates(entities)

for input_text in inputs:
    entities = extract_entities(input_text)

all_entities = []
for i, input_text in enumerate(inputs):
    entities = extract_entities(input_text)
    relevance_cargo = check_for_relevance(entities, input_text)
    filter1 = remove_singular_quantities(relevance_cargo)
    filter2 = remove_plural_weights(filter1)
    result = add_missing_weights(filter2)
    all_entities.append(result)

    print("------------------------------------------------------------------------------")
    print(f"Input text ({i+1}) = {input_text}")
    print(" ")
    print(f"Entities extracted = {result}")
    print("------------------------------------------------------------------------------")

data = {
    'Input Text': inputs,  # Ensure this matches the length of all_entities
    'Cargo': [', '.join(ent.get('Cargo', [])) for ent in all_entities],
    'Packaging': [', '.join(ent.get('Packaging', [])) for ent in all_entities],
    'Quantity': [],
    'Containment': [],
    'Weight': [],
    'Hazardous Properties': [ent.get('Hazardous properties', False) for ent in all_entities],
    'True Positives (TP)': '',  # Empty for manual entry
    'False Positives (FP)': '',  # Empty for manual entry
    'False Negatives (FN)': ''  # Empty for manual entry
}

for ent in all_entities:
    quantity_details = ', '.join(f"{key}: {value}" for key, value in ent.items() if 'quantity_' in key)
    data['Quantity'].append(quantity_details)

    containment_details = ', '.join(f"{key}: {value}" for key, value in ent.items() if
                                    key.endswith('_contain'))
    data['Containment'].append(containment_details)

    # Extracting weights dynamically
    weight_get = [', '.join(ent.get('Weight', [])) for ent in all_entities]
    weight_details = ', '.join(f"{key}: {value}" for key, value in ent.get('Weight').items() if
                               key.startswith('weight_per_'))
    data['Weight'].append(weight_details)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df = pd.DataFrame(data)


csv_filename = f'./data/Entities_Extracted - {timestamp}.csv'
df.to_csv(csv_filename, index=False)

df, csv_filename

excel_writer = pd.ExcelWriter(f'Entities_Extracted - {timestamp}.xlsx', engine='xlsxwriter')

df.to_excel(excel_writer, sheet_name='experiment1', index=False)

workbook  = excel_writer.book
worksheet = excel_writer.sheets['experiment1']

header_format = workbook.add_format({
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BC',
    'border': 1
})

body_format = workbook.add_format({
    'text_wrap': True,
    'valign': 'top',
    'border': 1,
    'indent': 1
})

border_format = workbook.add_format({
    'border': 1,
    'border_color': 'black',
    'fg_color': '#d5d7e3',
    'align': 'center',
    'valign': 'vcenter',
    'font_size': 14
})


for col_num, value in enumerate(df.columns.values):
    worksheet.write(0, col_num, value, header_format)


for row_num, data in enumerate(df.values, 1):
    for col_num, value in enumerate(data):
        worksheet.write(row_num, col_num, value, body_format)


for row_num, data in enumerate(df.values, 1):
    for col_num, value in enumerate(data):

        worksheet.write(row_num, col_num, value, body_format)

        if col_num in (7, 8, 9):
            worksheet.write(row_num, col_num, value, border_format)


worksheet.set_column('A:A', 60)
worksheet.set_column('B:B', 13)
worksheet.set_column('C:C', 13)
worksheet.set_column('D:D', 23)
worksheet.set_column('E:E', 23)
worksheet.set_column('F:F', 23)
worksheet.set_column('G:G', 10)
worksheet.set_column('H:H', 8.5)
worksheet.set_column('I:I', 8.5)
worksheet.set_column('J:J', 8.5)

worksheet.freeze_panes(1, 0)

excel_writer.close()
