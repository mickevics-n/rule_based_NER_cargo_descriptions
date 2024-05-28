import random
import sys
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats

primary_file_path = './data/Primary.csv'
secondary_file_path = './data/Secondary.csv'
numeric_codes_path = './data/Numeric codes.csv'

df_primary = pd.read_csv(primary_file_path)
df_secondary = pd.read_csv(secondary_file_path)
df_numeric_codes = pd.read_csv(numeric_codes_path)

def get_volume_range(package_type_code, df_numeric_codes):
    package_info = df_numeric_codes[df_numeric_codes['Package_type_code'] == package_type_code]
    if not package_info.empty:
        min_weight = package_info['Min_weight_kg'].values[0]
        max_weight = package_info['Max_weight_kg'].values[0]
        min_volume = package_info['Min_volume_m3'].values[0]
        max_volume = package_info['Max_volume_m3'].values[0]
        return min_weight, max_weight, min_volume, max_volume
    else:
        return None, None, None, None

def choose_value_between_min_max(primary_row):

    numeric_code = primary_row['Package_type_code']

    _, __, min_volume, max_volume = get_volume_range(numeric_code, df_numeric_codes)

    mean = (min_volume + max_volume) / 2
    std_dev = (max_volume - min_volume) / 6  # Approximation for std deviation

    chosen_value = np.random.normal(mean, std_dev)

    chosen_value = max(min_volume, min(max_volume, chosen_value))

    chosen_value = round(chosen_value, 4)

    return chosen_value

def generate_packaging_dimensions(volume, packaging_type):
    default_coefficients = {
        'AM': (1, 1, 3),
        'AT': (1, 1, 3),
        'BG': (3, 1, 3),
        'BB': (1, 1, 1),
        'BO': (1, 1, 3),
        'JR': (1, 1, 2),
        'CI': (1, 1, 2),
        'CA': (1, 1, 3),
        'BJ': (1, 1, 2)
    }

    if packaging_type in default_coefficients:
        coefficients = default_coefficients[packaging_type]

        base_volume = volume / (coefficients[0] * coefficients[1] * coefficients[2])
        side_length = round(base_volume ** (1 / 3), 2)

        dimensions = (
            round(side_length * coefficients[0], 2),
            round(side_length * coefficients[1], 2),
            round(side_length * coefficients[2], 2)
        )
    else:

        side_length = round(volume ** (1 / 3), 2)
        dimensions = (
            round(side_length * np.random.uniform(0.8, 1.2), 2),
            round(side_length * np.random.uniform(0.8, 1.2), 2),
            round(volume / (side_length * np.random.uniform(0.8, 1.2)), 2)
        )

    return dimensions


def calculate_weight_based_on_volume(volume, min_volume, max_volume, min_weight, max_weight):

    volume_percentile = (volume - min_volume) / (max_volume - min_volume)

    weight = min_weight + volume_percentile * (max_weight - min_weight)

    weight_round = round(weight, 0)
    if weight_round <= 0.0:
        return round(weight, 4)
    elif weight_round == np.nan:
        print("ValueError: cannot convert float NaN to integer. Run code again")
        sys.exit(1)
    else:
        return int(weight_round)


def calculate_number_of_primary_packages(primary_dimensions, primary_weight, secondary_row):
    primary_volume = np.prod(primary_dimensions)
    numeric_code = secondary_row['Package_type_code']

    min_weight_secondary, max_weight_secondary, min_volume_secondary, max_volume_secondary = get_volume_range(
        numeric_code, df_numeric_codes)

    max_volume_secondary *= 0.9

    volume_dist = stats.skewnorm(a=5, loc=min_volume_secondary, scale=(max_volume_secondary - min_volume_secondary))
    chosen_volume = volume_dist.rvs(1)[0]
    chosen_volume = min(max_volume_secondary, max(min_volume_secondary, chosen_volume))

    weight_dist = stats.skewnorm(a=5, loc=min_weight_secondary, scale=(max_weight_secondary - min_weight_secondary))
    chosen_weight = weight_dist.rvs(1)[0]
    chosen_weight = min(max_weight_secondary, max(min_weight_secondary, chosen_weight))

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    try:
        max_packages_by_volume = chosen_volume / primary_volume
    except RuntimeWarning:
        print("Warning: Divide by zero encountered in primary volume calculation")
        sys.exit(1)
    try:
        max_packages_by_weight = chosen_weight / primary_weight
    except RuntimeWarning:
        print("Warning: Divide by zero encountered in primary weight calculation")
        sys.exit(1)

    num_primary_packages = min(max_packages_by_volume, max_packages_by_weight)

    num_primary_packages = int(np.floor(num_primary_packages))

    secondary_weight = int(np.floor(num_primary_packages * primary_weight * 1.1))
    secondary_volume = chosen_volume

    return num_primary_packages, secondary_weight, round(secondary_volume, 4)

def choose_random_cargo_row_based_on_secondary():

    random_primary_row = df_primary.sample(n=1).iloc[0]
    cargo_types_in_row = random_primary_row['Cargo types'].split(', ')
    random_cargo_type_in_row = random.choice(cargo_types_in_row)
    chosen_volume_temp = choose_value_between_min_max(random_primary_row)
    packaging_dimensions = generate_packaging_dimensions(chosen_volume_temp, random_primary_row['Packaging type'])
    chosen_volume_temp_list = list(packaging_dimensions)
    chosen_volume = round(chosen_volume_temp_list[0] * chosen_volume_temp_list[1] * chosen_volume_temp_list[2], 4)
    numeric_code = random_primary_row['Package_type_code']

    min_weight, max_weight, min_volume, max_volume = get_volume_range(numeric_code, df_numeric_codes)

    chosen_weight = calculate_weight_based_on_volume(chosen_volume, min_volume, max_volume, min_weight, max_weight)

    packaging_type_primary = random_primary_row['Packaging type']

    filtered_secondary_df = df_secondary[df_secondary['comparable_primary'].str.contains(packaging_type_primary)]

    if not filtered_secondary_df.empty:

        random_secondary_row = filtered_secondary_df.sample(n=1).iloc[0]

        quantity_primary, secondary_weight, secondary_volume = calculate_number_of_primary_packages(
            packaging_dimensions, chosen_weight, random_secondary_row.iloc[2:-1].to_dict())

        max_weight_pallet = 1500
        min_weight_pallet = 1100
        min_volume_pallet = 2
        max_volume_pallet = 4 * 0.9

        volume_dist = stats.skewnorm(a=8, loc=min_volume_pallet, scale=(max_volume_pallet - min_volume_pallet))
        secondary_chosen_volume = volume_dist.rvs(1)[0]
        secondary_chosen_volume = min(max_volume_pallet, max(min_volume_pallet, secondary_chosen_volume))

        weight_dist = stats.skewnorm(a=8, loc=min_weight_pallet, scale=(max_weight_pallet - min_weight_pallet))
        secondary_chosen_weight = weight_dist.rvs(1)[0]
        secondary_chosen_weight = min(max_weight_pallet, max(min_weight_pallet, secondary_chosen_weight))

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        try:
            max_packages_by_volume = secondary_chosen_volume / secondary_volume
        except RuntimeWarning:
            print("Warning: Divide by zero encountered in secondary volume calculation")
            sys.exit(1)

        try:
            max_packages_by_weight = secondary_chosen_weight / secondary_weight
        except RuntimeWarning:
            print("Warning: Divide by zero encountered in secondary weight calculation")
            sys.exit(1)

        # The number of packages is constrained by both volume and weight
        num_secondary_packages = min(max_packages_by_volume, max_packages_by_weight)

        # Round down to ensure it fits
        num_secondary_packages = int(np.floor(num_secondary_packages))

        # Store the entire row and the additional random cargo type in a dictionary
        result = {
            'Cargo type': random_cargo_type_in_row,
            'Primary packaging type': random_primary_row['Description'],
            'Primary packaging quantity': quantity_primary,
            'Primary packaging weight': chosen_weight,
            'Secondary packaging type': random_secondary_row['Description'],
            'Secondary packaging weight': secondary_weight,
            'Secondary packaging quantity': num_secondary_packages

        }

        return result
    else:

        max_weight_pallet = 1500
        min_weight_pallet = 1100
        min_volume_pallet = 2
        max_volume_pallet = 4 * 0.9

        volume_dist = stats.skewnorm(a=8, loc=min_volume_pallet, scale=(max_volume_pallet - min_volume_pallet))
        tertiary_chosen_volume = volume_dist.rvs(1)[0]
        tertiary_chosen_volume = min(max_volume_pallet, max(min_volume_pallet, tertiary_chosen_volume))

        weight_dist = stats.skewnorm(a=8, loc=min_weight_pallet, scale=(max_weight_pallet - min_weight_pallet))
        tertiary_chosen_weight = weight_dist.rvs(1)[0]
        tertiary_chosen_weight = min(max_weight_pallet, max(min_weight_pallet, tertiary_chosen_weight))

        # Calculate the number of primary packages that can fit within the chosen volume and weight
        max_packages_by_volume = tertiary_chosen_volume / chosen_volume
        max_packages_by_weight = tertiary_chosen_weight / chosen_weight

        # The number of packages is constrained by both volume and weight
        num_primary_packages = min(max_packages_by_volume, max_packages_by_weight)

        # Round down to ensure it fits
        num_primary_packages = int(np.floor(num_primary_packages))

        result = {
            'Cargo type': random_cargo_type_in_row,
            'Primary packaging type': random_primary_row['Description'],
            'Primary packaging quantity': num_primary_packages,
            'Primary packaging weight': chosen_weight,
            'Secondary packaging type': None,
            'Secondary packaging weight': None,

        }
        return result

def generate_chatgpt_prompt(result):
    pallet_quantity = [10, 11, 21, 22, 23]
    random_number = random.choice(pallet_quantity)
    primary_package = result['Primary packaging type']
    secondary_package = result['Secondary packaging type']

    per_str = "per"
    if result['Secondary packaging type'] == None:
        per_str = ""
        secondary_package = ""
    prompt = (
        "You are tasked with generating a synthetic customer request e-mail containing a cargo description based on the following details:\n\n"

        f"- **Cargo type:** {result['Cargo type']}\n"
        f"- **Primary packaging:** {result['Primary packaging type']}\n"
        f"- **Total number of {primary_package} {per_str} {secondary_package}:** {result['Primary packaging quantity']}\n"
        f"- **Weight of each {primary_package}:** {result['Primary packaging weight']}\n"
        f"- {result['Primary packaging quantity']} {primary_package} are placed on a pallet. Total number of pallets is {random_number}\n"
    )

    if result.get('Secondary packaging type') and result.get('Secondary packaging weight') and result.get(
            'Secondary packaging quantity'):
        lines = prompt.split('\n')

        # Remove the last non-empty line
        lines = [line for line in lines if line.strip() != ""]
        lines.pop()

        # Join the lines back into a string
        prompt = '\n'.join(lines)
        prompt += ("\n"
            f"- **Secondary packaging:** {result['Secondary packaging type']}\n"
            f"- **Total number of {secondary_package} per pallet:** {result['Secondary packaging quantity']}\n"
            f"- **Weight of each {secondary_package}:** {result['Secondary packaging weight']}\n"
            f"- {result['Secondary packaging quantity']} {secondary_package} are placed on a pallet. Total number of pallets is {random_number}\n"
        )

    prompt += (
        "\n### Instructions:\n"
        "1. Describe the cargo type and its packaging naturally, as a person would. Use either numeric or word formats for numbers.\n"
        "2. Do not use technical terms like primary and secondary packaging types when generating e-mail.\n"
        "3. The weight values provided are in kilograms (kg), but you may convert them into tonnes, grams, or pounds as appropriate.\n"
        "4. The description should flow naturally without using bullet points or summarizing the packaging information.\n"
        "5. Clearly state the quantity and weight of the packaged items.\n"
        "6. If the cargo type can be regarded as hazardous, clearly state its hazardous properties.\n"
        "7. Vary the sentence structure and wording to ensure the description is engaging and clear.\n"
        "8. Add random Recipient's Name, Position, and Company Name.\n"
    )
    print(prompt)
    return prompt


result = choose_random_cargo_row_based_on_secondary()
generate_chatgpt_prompt(result)