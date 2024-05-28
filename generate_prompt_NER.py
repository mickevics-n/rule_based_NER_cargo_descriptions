import random
import pandas as pd

primary_file_path = './data/Primary.csv'

df_primary = pd.read_csv(primary_file_path)

def get_cargo_and_packaging_type(df_primary):

    random_primary_row = df_primary.sample(n=1).iloc[0]

    packaging_type = random_primary_row['Description']

    cargo_types_in_row = random_primary_row['Cargo types'].split(', ')

    cargo_type = random.choice(cargo_types_in_row)

    return packaging_type, cargo_type

def compose_knowledge_base(df_primary):
    knowledge_base = {}
    for i in range(25):
        packaging_type, cargo_type = get_cargo_and_packaging_type(df_primary)
        key = f"Possible combination {i+1}"
        knowledge_base[key] = f"cargo_type: {cargo_type}, packaging_type: {packaging_type}"

    return knowledge_base

def generate_chatgpt_prompt(df_primary):
    knowledge_base = compose_knowledge_base(df_primary)

    description_examples = [
        "One-hundred-and-fifty bags of bolts, each containing 100 pieces. Each bag with bolts weighs 2 kg. These bolts are essential for assembling machinery in the factory.",
        "Forty drums of industrial oil, each containing 200 liters. Total weight is 8 tonnes.",
        "Twenty-four wooden barrels of cold-pressed olive oil, each weighing 180 kg. The olive oil is stored under organic certification standards, and the barrels are constructed from sustainably sourced oak. This shipment does not fall under any hazardous materials classification",
        "Five bags of flammable chemicals, packed securely in fireproof containers. Each bag weighs 20 pounds and is marked as hazardous.",
        "Three pallets of paint, carefully packed to prevent spillage during transit. Each pallet contains 50 buckets of paint. Each bucket of paint weighs 1 kg. ",
        "Fifteen plastic pallets of medical supplies, each weighing 900 kilograms and containing 88 pieces of supplies. The supplies include bandages, syringes, and medications. All items are packed in boxes weighing 5 pounds each.",
    ]

    prompt = (
        f"\nDescription examples = {description_examples}\n"
    )

    prompt += (
        f"\nknowledge_base = {knowledge_base}\n"
    )

    prompt += (
        "\nUsing the provided knowledge base, generate single or multi-sentence 25 cargo descriptions. The descriptions should include a cargo type, a packaging type, and a weight type, all of which must match the allowed types in the knowledge base. In some cases, information might be missing from a description. In some cases, the description could include irrelevant information that still uses any of the provided cargo types. Twenty percent of the cargo descriptions must describe hazardous cargo, stating its properties. Twenty-five percent of the cargo descriptions must use weight units different from kg/kilograms.\n"
    )

    prompt += (
        "\nFormat generated descriptions in copy/paste friendly format:\n"
        "inputs = [ ]"
    )

    print(prompt)
    return prompt

generate_chatgpt_prompt(df_primary)