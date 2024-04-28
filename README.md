# Cargo Information Extraction (rule-based NER)

## Description
This project is designed to process textual descriptions of cargo and extract detailed entities such as cargo types, packaging, weights, and quantities. It utilizes natural language processing techniques and regular expressions to identify and categorize information, which is crucial for logistics, inventory management, and shipping industries.

## Installation
To run this project, you will need Python 3.8 or higher and the following Python packages:
- `re` for regular expressions
- `nltk` for natural language processing
- `word2number` for converting words to numbers

You can install the necessary dependencies via pip:
```bash
pip install nltk word2number
```

## Usage

Cargo description examples are stored in `inputs.py` and are formatted as a list of strings. New descriptions can be generated using OpenAI's ChatGPT-3.5 by employing the following prompt:

```plaintext
description_examples = [
    "One-hundred-and-fifty bags of bolts, each containing 100 pieces. Each bag with bolts weighs 2 kg. These bolts are essential for assembling machinery in the factory.", 
    "Twenty-four wooden barrels of cold-pressed olive oil, each weighing 180 kg. The olive oil is stored under organic certification standards, and the barrels are constructed from sustainably sourced oak. This shipment does not fall under any hazardous materials classification",
    "Five bags of flammable chemicals, packed securely in fireproof containers. Each bag weighs 20 pounds and is marked as hazardous.",
    "Three pallets of paint, carefully packed to prevent spillage during transit. Each pallet contains 50 buckets of paint. Each bucket of paint weighs 1 kg. ",
    "Fifteen plastic pallets of medical supplies, each weighing 900 kilograms. The supplies include bandages, syringes, and medications. All items are packed in boxes weighing 5 pounds each.”,
    ]

cargo_types = [
    "olive oil",
    "apples",
    "ceramic pottery",
    "spirits",
    "industrial oil",
    "bolts",
    "machinery",
    "petroleum",
    "paint",
    "equipment",
    "chemicals",
    "textiles",
    "automobile parts",
    "engine components",
    "electronics",
    "furniture",
    "plastics",
    "medical supplies",
    "steel",
    "wood products",
    "pharmaceuticals",
    "raw materials",
    "construction materials",
    "cosmetics",
    "alcoholic beverages",
    "glassware",
    "toys",
    "agricultural products",
    "luxury goods",
    "stationery",
    "sporting goods",
    "packaging materials",
    "household goods",
    "food products",
    "perfumes",
    "clothing",
    "jewelry",
    "musical instruments",
    "artwork",
    "office supplies"
]

packaging_types = [
    "drums",
    "containers",
    "tubs",
    "cans",
    "vials",
    "packs",
    "sacks",
    "tanks",
    "cases",
    "packets",
    "crates",
    "casks",
    "totes",
    "barrels",
    "jugs",
    "flasks",
    "envelopes",
    "bins",
    "canisters",
    "barrels",
    "tins",
    "drums",
    "bags",
    "bottles",
    "packets",
    "crates",
    "jars",
    "boxes",
    "pallets",
]
weight type = [
     "kg",
     "kilograms",
     "tonnes",
     "pounds",
     "tonne",
     "kilogram",
     "pound",
     "gram",
     "grams",
]

Using the provided knowledge base, generate 25 cargo descriptions. The descriptions should include a cargo type, a packaging type, and a weight type, all of which must match the allowed types in the knowledge base. In some cases, information might be missing from a description. In some cases, the description could include irrelevant information that still uses any of the provided types. Use dashed format when expressing numbers as words. Twenty percent of the cargo descriptions must describe hazardous cargo.

Format generated descriptions in copy/paste friendly format: 

inputs = [
] 
```
After generating the descriptions, copy and paste them into the `inputs.py` file, replacing the existing content of the `inputs` variable.

To run the `extract_entities.py` script and print the extracted entities directly to the terminal, open your command prompt or terminal and execute the following command:

```bash
python extract_entities.py
```

### Post-Installation Adjustment

After you have successfully run the `extract_entities.py` file for the first time, it's recommended to comment out the following lines in the `scripts.py` file. These lines involve downloading NLTK resources that are required only on the initial run. Commenting them out will prevent unnecessary re-downloading in future executions, which can save time and network resources.

```python
# Comment these lines after the first execution
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
```
## Input Format

The function expects a list of strings, where each string contains detailed descriptions of different cargo items. These descriptions include crucial information such as the weight of the cargo and the type of packaging used. Below is an example of how to format the inputs for the function:

```python
inputs = [
    "A string containing a description of cargo, including details about weight and packaging.",
    "Another string describing different cargo with specific details."
]
```

## Output Format

When the `extract_entities.py` script is executed, the extracted entities are printed to the terminal. The output provides a structured dictionary that categorizes and lists various attributes related to the cargo. Here is the breakdown of the output format:

- **Cargo**: Lists the types of cargo identified in the input.
- **Packaging**: Shows the types of packaging noted in the cargo description.
- **quantity_\<packaging type\>**: Indicates the quantity associated with a specific type of packaging.
- **\<packaging type\>_contain**: Describes what each type of packaging contains, if applicable.
- **Weight**: Provides a dictionary where each key starts with `weight_per_` followed by the packaging type and the value is the weight of that packaging type in kilograms.
- **Hazardous properties**: Boolean value indicating whether the cargo has hazardous properties.

### Example Output

When the `extract_entities.py` script is executed with a specific text input, the terminal displays the extracted entities in a structured format. Below is an example of how the output is formatted based on the following input text:

**Input Text:**
"Ten barrels of spirits, each filled with 200 liters of premium whiskey. Each barrel weighs 250 kilograms and is made from charred oak to enhance flavor."

**Output in Terminal:**
```plaintext
{
    'Cargo': ['spirits'],
    'Packaging': ['barrels', 'barrel'],
    'quantity_barrels': '10',
    'barrel_contain': '200 liters',
    'Weight': {'weight_per_barrel': 250.0},
    'Hazardous properties': False
}
```

