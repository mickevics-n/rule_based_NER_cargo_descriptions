import pandas as pd

primary_file_path = 'C:/Users/micke/PycharmProjectsNER/data/Primary.csv'
secondary_file_path = 'C:/Users/micke/PycharmProjectsNER/data/Secondary.csv'
df_primary = pd.read_csv(primary_file_path)
df_secondary = pd.read_csv(secondary_file_path)

def extract_cargo_types(df_primary):

    df_primary['Cargo types'] = df_primary['Cargo types'].str.split(', ')
    unique_cargo_types = set(df_primary['Cargo types'].sum())
    unique_cargo_types_list = list(unique_cargo_types)

    return unique_cargo_types_list

def extract_packaging_types(df_primary, df_secondary):

    df_primary['Description'] = df_primary['Description'].str.split(', ')
    df_secondary['Description'] = df_secondary['Description'].str.split(', ')

    unique_primary_packaging_types = set(df_primary['Description'].sum())
    unique_secondary_packaging_types = set(df_secondary['Description'].sum())

    combined_packaging_types_set = unique_primary_packaging_types | unique_secondary_packaging_types

    unique_packaging_types_list = list(combined_packaging_types_set)

    return unique_packaging_types_list

def extract_packaging_short_form(df_primary, df_secondary):
    df_primary['Packaging Type'] = df_primary['Packaging Type'].str.split(', ')
    df_secondary['Packaging Type'] = df_secondary['Packaging Type'].str.split(', ')

    unique_primary_packaging_types = set(df_primary['Packaging Type'].sum())
    unique_secondary_packaging_types = set(df_secondary['Packaging Type'].sum())

    combined_packaging_types_set = unique_primary_packaging_types | unique_secondary_packaging_types

    unique_packaging_short_list = list(combined_packaging_types_set)

    return unique_packaging_short_list


cargo_items = extract_cargo_types(df_primary)

packaging_plural = extract_packaging_types(df_primary, df_secondary)

packaging_short = extract_packaging_short_form(df_primary, df_secondary)

