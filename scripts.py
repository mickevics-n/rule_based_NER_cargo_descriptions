import re
from word2number import w2n
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from knowlege_base import packaging_plural

# Comment lines when downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def compose_packaging_pattern(packaging_plural):
    """
    Constructs a regex pattern used to extract packaging types from text and generates a list of packaging types
    in both plural and singular forms. The function takes a list of plural packaging types as input and uses the
    WordNetLemmatizer from NLTK to convert each to its singular form. Both the original plural terms and the derived
    singular forms are added to a new list.

    Args:
        packaging_plural (list): A list containing the plural forms of various packaging types, serving as a knowledge
        base.

    Returns:
        tuple: A tuple containing two elements:
            - packaging_pattern (str): A regex pattern designed to recognize the listed packaging terms in text.
            - packaging (list): An updated list of packaging types, including both original plural forms and the
            created singular forms.
    """
    lemmatizer = WordNetLemmatizer()

    packaging = []
    for term in packaging_plural:
        packaging.append(term)
        singular = ' '.join([lemmatizer.lemmatize(word, 'n') for word in term.split()])
        if singular not in packaging:
            packaging.append(singular)

    packaging_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in packaging) + r')\b'

    return packaging_pattern, packaging

def is_singular(word):
    """
    Determines whether a given word is singular using NLTK's part-of-speech (POS) tagging.
    This function first checks predefined lists of exceptions to quickly determine singularity or plurality without
    the need for POS tagging. If the word is not listed in either exception list, the function proceeds with NLTK's
    POS tagging to analyze the word's grammatical category. It specifically checks if the word is tagged as a
    singular noun ('NN'). Based on this tagging, the function returns True for singular nouns and False otherwise.

    Args:
        word (str): The word to be checked for singularity.

    Returns:
        bool: True if the word is identified as singular. Returns False if the word is identified as plural.
    """
    exceptions_plural = [
        'pallets',
        'Pallets',
    ]
    exceptions_singular = [
        'can',
        'Can',
    ]

    if word in exceptions_plural:
        return False

    if word in exceptions_singular:
        return True

    tagged_word = nltk.pos_tag([word])
    pos = tagged_word[0][1]

    return pos == 'NN'


def to_singular(word):
    """
    Converts a given plural word to its singular form using NLTK's WordNet lemmatization capabilities. Initially, the
    function checks if the input word is already singular by utilizing a helper function `is_singular`.
    If the word is determined to be singular, it is returned as is. The function also incorporates a predefined
    exceptions list to handle specific common or irregular plurals directly. WordNet is used to attempt a general
    conversion by identifying the first lemma of the word that differs from the original input, which is assumed to
    be its singular form. If no suitable singular form is identified, the original word is returned unchanged.

    Args:
        word (str): The word to be converted into its singular form.

    Returns:
        str: The singular form of the input word. If the word is already singular, the input word is returned.
    """

    # Check if the word is already singular
    if is_singular(word):
        return word

    # Define exceptions for specific words
    exceptions = {
        'pallets': 'pallet',
        'Pallets': 'Pallet',
        'pallet': 'pallet',
        'Pallet': 'Pallet',
    }

    # Check for exceptions
    if word in exceptions:
        return exceptions[word]

    # Use WordNet for regular conversion
    synsets = wordnet.synsets(word)
    if synsets:
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    return lemma.name()
    return word


def extract_quantities(input_text):
    """
    Extracts quantities related to packaging from a given text. This function scans the text for predefined patterns
    associated with packaging types and retrieves numeric values that precede these packaging identifiers.
    The function is designed to handle different formulations, where the quantity may be immediately before the
    packaging type or separated by up to two intervening words. If a packaging type is identified as singular without
    any numeric descriptor, it is skipped. In cases where no explicit quantity is mentioned for a packaging type, it
    defaults to "0".

    Args:
        input_text (str): A string containing a description of cargo, including details about weight and packaging.

    Returns:
        list: A list of tuples, each containing a quantity and its associated packaging type (e.g., [('13', 'bins')]).
    """
    packaging_pattern, _ = compose_packaging_pattern(packaging_plural)

    packaging_matches = re.findall(packaging_pattern, input_text)

    quantity_matches = []
    for package in packaging_matches:

        if is_singular(package):
            #print(f"Packaging {package} is recognised as singular: abort")
            continue

        pattern = r"\b(\d+)\s+(?:\w+\s+)?(?:\w+\s+)?" + re.escape(package)
        matches = re.findall(pattern, input_text)

        if matches:
            quantity_matches.extend([(match, package) for match in matches])
        else:
            quantity_matches.append(("0", package))

    return quantity_matches

def convert_to_kg(weight, unit):
    """
    Converts a given weight measurement from various units to kilograms.  If the unit is already in kilograms, it
    returns the weight as is.

    Args:
        weight (float): The numerical value of the weight to be converted.
        unit (str): The unit of measurement associated with the weight (e.g., 'kg', 'gram', 'tonne', 'pound').

    Returns:
        float: The weight converted into kilograms, returned as a string for consistency in data formatting.
        If the input unit is unrecognized, the function raises a ValueError indicating the issue.
    """
    if unit in ['tonnes', 'tonne']:
        return str(float(weight) * 1000)  # 1 tonne = 1000 kilograms
    elif unit in ['pounds', 'pound']:
        return str(float(weight) * 0.453592)  # 1 pound = 0.453592 kilograms
    elif unit in ['kg', 'kilograms', 'kilogram']:
        return weight
    elif unit in ['grams', 'gram']:
        return str(float(weight) / 1000)  # 1 gram = 0.001 kilograms
    else:
        raise ValueError("Invalid unit. Unit matched different from 'kg', 'kilograms', 'tonnes', 'pounds', or 'grams'.")

def count_words_until_match_left(pakaging, full_weight, input_text):
    """
    Counts words in between pakaging and full_weight in input_text.

    Args:
        pakaging (str): a string containing packaging info.
        full_weight (str) a string containing weight + unit info.
        input_text (str) a string containing a description of cargo, including details about weight and packaging.

    Returns:
        int: count
    """
    start_index = input_text.find(full_weight)
    substring_before_weight = input_text[:start_index]
    reversed_substring = substring_before_weight[::-1]
    words = reversed_substring.split()
    count = 0
    reversed_case = pakaging[::-1]
    for word in words:
        if reversed_case in word:
            break
        count += 1

    return count

def filter_data_by_recognised(data, recognised_weights):
    """
    Filters recognised weight values based on values calculated in count_words_until_match_left function.

    Args:
        data (dict): A dictionary containing information containing recognised weight(s), unit, and assigned score. e.g:
        data = {
                    'cases': [['15', 'kilograms', 10], []]
                    'boxes': [['15', 'kilograms', 3]], []]
                    'case': [['15', 'kilograms', 1]], []]
                    'box': [['15', 'kilograms', 3], ['14', 'pounds', 1]]
                }
        recognised_weights (list) A list containing tuples with recognised weight and unit for reference. e.g:
        recognised_weights =
        [('15', 'kilograms'), ('14', 'pounds')]

    Returns:
        dict: Sorted dictionary with lowest scores. e.g:
        data = {
                    'case': [['15', 'kilograms', 1]]
                    'box': [['14', 'pounds', 1]]
                }
    """

    min_values = {weight_unit: float('inf') for weight_unit in recognised_weights}

    keys_to_remove = [key for key, value in data.items() if not any(value)]

    for key in keys_to_remove:
        del data[key]

    for value_lists in data.values():
        for sublist in value_lists:
            if len(sublist) == 3 and tuple(sublist[:2]) in recognised_weights:
                current_weight_unit = tuple(sublist[:2])
                current_value = sublist[2]
                if current_value < min_values[current_weight_unit]:
                    min_values[current_weight_unit] = current_value
            elif len(sublist) == 4 and (sublist[0], sublist[2]) in recognised_weights:
                current_weight_unit = (sublist[0], sublist[2])
                current_value = sublist[3]
                if current_value < min_values[current_weight_unit]:
                    min_values[current_weight_unit] = current_value

    new_data = {}
    for key, value_lists in data.items():
        for sublists in value_lists:

            if len(sublists) == 3:
                filtered_lists = [
                    sublist[:2]
                    for sublist in value_lists
                    if len(sublist) == 3 and tuple(sublist[:2]) in recognised_weights and sublist[2] == min_values[
                        tuple(sublist[:2])]
                ]
            elif len(sublists) == 4:

                filtered_lists_pop = [
                    sublist[:3]
                    for sublist in value_lists
                    if len(sublist) == 4 and
                       (sublist[0], sublist[2]) in recognised_weights and
                       sublist[3] == min_values[(sublist[0], sublist[2])]
                ]
                filtered_lists = [
                    [sublist[1], sublist[2]]
                    for sublist in filtered_lists_pop
                ]
            elif len(sublists) == 0:
                continue

        if len(filtered_lists) == 1:
            new_data[key] = filtered_lists[0]
        elif len(filtered_lists) > 1:
            unique_items = dict.fromkeys(item for sublist in filtered_lists for item in sublist).keys()
            new_data[key] = list(unique_items)

    return new_data

def extract_weight_in_kg(input_text, packaging_quantities):
    """
    Extracts weight measurements from a given input_text that describes cargo, and converts these weights into kilograms
    where necessary with convert_to_kg(weight, unit) function.

    Args:
        input_text (str): A string containing a description of cargo, including details about weight and packaging.

    Returns:
        dict: A dictionary with keys representing each packaging type found in the text appended with 'weight_per_',
        and values being the corresponding weight in kilograms. If the weight information is missing or incomplete for
        any package, the value is set to 'information missing'.
    """

    packaging_pattern, _ = compose_packaging_pattern(packaging_plural)

    packaging_matches = re.findall(packaging_pattern, input_text)

    weight_pattern = r"\b(\d{1,3}(?:,\d{3})*\.\d+|\d{1,3}(?:,\d{3})*|\d+\.\d+|\d+)\s*(kg|kilograms|tonnes|pounds|tonne|kilogram|pound|gram|grams)\b"
    weight_units_pattern = r'(?:kg|kilograms|tonnes|pounds|tonne|kilogram|pound|grams|gram)'

    weights_sentence = re.findall(weight_pattern, input_text)

    weight_per_package = {package: [] for package in packaging_matches}

    sentences = re.split(r'(?<=[.!?]) +', input_text)
    list_weights = []
    list_units = []
    for sentence in sentences:

        for weight, unit in weights_sentence:

            if weight in sentence:

                for package in packaging_matches:
                    inner_weights = []
                    weight_per_package[package].append(inner_weights)
                    if package in sentence:
                        pattern = r'\b(?:with a total weight of|total of|total of|total weight is|total weight|Total weight is) (\d{1,3}(?:,\d{3})*\.\d+|\d{1,3}(?:,\d{3})*|\d+\.\d+|\d+) (' + weight_units_pattern + r')\b'
                        matches = list(re.finditer(pattern, input_text, re.IGNORECASE))

                        if 'each' in sentence or 'Each' in sentence:
                            inner_weights.append(weight)
                            inner_weights.append(unit)
                            joined_weights = ' '.join([weight, unit])
                            count = count_words_until_match_left(package, joined_weights, input_text)
                            inner_weights.append(count)
                            list_weights.append(weight)
                            list_units.append(unit)

                        elif not matches:

                            inner_weights.append(weight)
                            inner_weights.append(unit)
                            joined_weights = ' '.join([weight, unit])
                            count = count_words_until_match_left(package, joined_weights, input_text)
                            inner_weights.append(count)
                            list_weights.append(weight)
                            list_units.append(unit)

                        elif matches:

                            for match in matches:
                                for key in packaging_quantities:
                                    if package in key:
                                        value = packaging_quantities[key]
                                        joined_weights = ' '.join([weight, unit])
                                        count = count_words_until_match_left(package, joined_weights, input_text)
                                        if value == 0 or value == 'information missing':
                                            continue
                                        else:
                                            inner_weights.append(weight)
                                            inner_weights.append(float(float(weight) / float(value)))
                                            inner_weights.append(unit)
                                            inner_weights.append(count)
                                            list_weights.append(float(float(weight) / float(value)))
                                            list_units.append(unit)
                                    else:
                                        continue

                        else:
                            continue

                    else:
                        pattern = r'\b(?:total of|total weight is|total weight|Total weight is) (\d{1,3}(?:,\d{3})*\.\d+|\d{1,3}(?:,\d{3})*|\d+\.\d+|\d+) (' + weight_units_pattern + r')\b'
                        matches = list(re.finditer(pattern, input_text, re.IGNORECASE))
                        for match in matches:
                            for key in packaging_quantities:

                                value = packaging_quantities[key]
                                if value == 0 or value == 'information missing':
                                    continue
                                joined_weights = ' '.join([weight, unit])
                                count = count_words_until_match_left(package, joined_weights, input_text)
                                inner_weights.append(weight)
                                inner_weights.append(float(float(weight) / float(value)))
                                inner_weights.append(unit)
                                inner_weights.append(count)
                                list_weights.append(float(float(weight) / float(value)))
                                list_units.append(unit)
            else:
                continue

    weight_per_package_filtered = filter_data_by_recognised(weight_per_package, weights_sentence)

    result = {}

    for package, weights in weight_per_package_filtered.items():

        word = to_singular(package)
        variable_name = f"weight_per_{word}"

        if len(weights) >= 2:
            weight = convert_to_kg(weights[0], weights[1])
            result[variable_name] = weight

        else:
            weight = "information missing"
            result[variable_name] = weight

    unique_weights_list = list(set(list_weights))
    unique_units_list = list(set(list_units))
    if len(result) == 1 and (weight_per_package) and len(unique_weights_list) == 1 and len(unique_units_list) == 1:
        weight = convert_to_kg(unique_weights_list[0], unique_units_list[0])
        result[variable_name] = weight

    return result


def add_commas(input_text):
    """
    Formats text by adding commas after numeric quantities followed by packaging types in cargo descriptions.

    Args:
        input_text (str): A string containing a description of cargo, including details about weight and packaging.

    Returns:
        modified_text (str): The input text, modified to include commas after each numeric quantity followed by a
        packaging type.
    """
    _, packaging = compose_packaging_pattern(packaging_plural)
    pattern = r"(\d+\s+(?:" + "|".join(map(re.escape, packaging)) + r")\b)"

    modified_text = re.sub(pattern, r'\1,', input_text)

    return modified_text



def word_to_num(word):
    """
    Converts numerical words to their corresponding integer representations. This function takes a single word as input
    and attempts to translate it into a numerical value. It leverages the `w2n.word_to_num` method from an external
    library, which is capable of understanding and converting words like 'two', 'twenty', or 'hundred' into their
    numeric equivalents (2, 20, 100).

    Args:
        word (str): The word to convert from text to a numeric value. The word must be a single numeric word or
        a compound word with components joined by dashes!

    Returns:
        int or None: Returns the numeric value of the word if it can be converted; otherwise, returns None.
    """
    try:
        return w2n.word_to_num(word)
    except ValueError:
        return None

def normalize_numbers_in_sentence(sentence):
    """
    Parses through a sentence and normalizes numbers expressed in words by combining consecutive numerical words with
    hyphens, including handling connector words like 'and' when they appear between numerical words.

    Args:
        sentence (str): The input sentence containing numerical words.

    Returns:
        str: A modified sentence where consecutive numerical words and certain connectors are joined by hyphens.
    """
    words = sentence.split()
    normalized_sentence = []
    current_number = []

    for word in words:
        if word == 'and' and current_number:

            try:
                next_index = words.index(word) + 1
                if next_index < len(words):
                    w2n.word_to_num(words[next_index])
                    current_number.append(word)
            except (ValueError, IndexError):

                if current_number:
                    normalized_sentence.append('-'.join(current_number))
                    current_number = []
                normalized_sentence.append(word)
        else:

            try:
                w2n.word_to_num(word)
                current_number.append(word)
            except ValueError:

                if current_number:
                    normalized_sentence.append('-'.join(current_number))
                    current_number = []
                normalized_sentence.append(word)

    if current_number:
        normalized_sentence.append('-'.join(current_number))

    return ' '.join(normalized_sentence)

def convert_numerical_words_to_numbers(input_text):
    """
    Converts all recognizable numerical words in a given text to their numeric equivalents. Also, checks if hyphen-separated
    words are all numerical before preserving the hyphen; otherwise, the hyphen is removed.

    Args:
        text (str): The input text containing potential numerical words.

    Returns:
        str: A new string where all numerical words have been replaced by their numeric equivalents and inappropriate hyphens are removed.
    """
    norm = normalize_numbers_in_sentence(input_text)
    words = norm.split(' ')

    processed_words = []

    for word in words:

        if '-' in word:
            subwords = word.split('-')

            if all(word_to_num(sub) is not None or sub == 'and' for sub in subwords):
                processed_word = '-'.join(str(sub) for sub in subwords)
            else:
                processed_words.extend(subwords)
            processed_words.append(processed_word)
        else:
            pattern = r"\b(\d{1,3}(?:,\d{3})*\.\d+|\d{1,3}(?:,\d{3})*|\d+\.\d+|\d+)\b"
            matches = re.findall(pattern, word)
            if matches:
                if ',' in word:
                    processed_word = word.replace(',', '')

                    processed_words.append(processed_word)
                elif '.' in word:
                    processed_word = str(word_to_num(word)) if word_to_num(word) is not None else word
                    processed_words.append(processed_word)
                else:
                    processed_word = str(word_to_num(word)) if word_to_num(word) is not None else word
                    processed_words.append(processed_word)
            else:

                processed_word = str(word_to_num(word)) if word_to_num(word) is not None else word
                processed_words.append(processed_word)



    num_converted_text = ' '.join(
        [str(word_to_num(word)) if word_to_num(word) is not None else word for word in processed_words])

    return num_converted_text

def is_hazardous(sentence):
    """
    Determines whether a given sentence suggests that the cargo discussed is hazardous. The function tokenizes the
    sentence to analyze its content word by word. It identifies any negation terms (such as 'not', 'no', 'n't',
    'never', 'none') and sets a flag if any are found. It then checks for the presence of keywords associated
    with hazardous conditions (like 'hazardous', 'dangerous', 'flammable', 'toxic', 'explosive').
    If a hazardous keyword is found, the presence of a prior negation modifies the interpretation:
    if a negation is present, the function returns False, indicating the cargo is not hazardous; otherwise,
    it returns True, indicating the cargo is hazardous. If no hazardous keywords are found,
    the function defaults to returning False, suggesting the cargo is non-hazardous.

    Args:
        sentence (str): A string containing the sentence to be analyzed.

    Returns:
        bool: Returns True if the sentence implies the cargo is hazardous, otherwise returns False.
    """

    tokens = word_tokenize(sentence)

    negations = set(['not', 'no', 'n\'t', 'never', 'none', 'non'])
    negation_flag = False
    for token in tokens:
        if token.lower() in negations:
            negation_flag = True
            break

    # Hazardous terms, could be modified
    hazardous_keywords = ['hazardous', 'hazard', 'dangerous', 'flammable', 'toxic', 'explosive']

    for word in hazardous_keywords:

        if word in tokens:
            if negation_flag:
                return False
            else:
                return True

    # Default to non-hazardous
    return False


def check_for_relevance(entities, input_text):
    """
    Checks the relevance of recognized cargo types within a text by analyzing their co-occurrence with other
    specified entities in the same sentence. The function preprocesses the input text to convert numerical words
    to their corresponding figures and assesses each sentence for the presence of cargo, weight, and packaging
    entities. It then counts the occurrences of all other relevant entities from the dictionary in these sentences.
    If a cargo type does not co-occur with other relevant entities, it is flagged as potentially irrelevant.

    Args:
        entities (dict): A dictionary containing lists of entities under keys such as "Cargo", "Weight", and
        "Packaging", alongside other relevant entity categories.

    Returns:
        dict: The updated dictionary with each cargo type that lacks significant related entity mentions marked as
        '{cargo} (might be irrelevant)' to indicate its potential irrelevance.
    """

    input_text = ' '.join(
        [str(word_to_num(word)) if word_to_num(word) is not None else word for word in input_text.split()])

    cargo_type = entities.get("Cargo")
    weight_type = entities.get("Weight")
    packaging_type = entities.get("Packaging")

    allentity_list = [value for key, value in entities.items() if key not in ["Cargo", "Weight", "Packaging"] and (isinstance(value, list) or not isinstance(value, bool))]

    sentences = re.split(r'(?<=[.!?]) +', input_text)

    relevancy_scores = {cargo: 0 for cargo in cargo_type}

    for sentence in sentences:

        for cargo in cargo_type:

                if cargo in sentence:

                    entity_count = {str(entity): sentence.count(str(entity)) for entity in allentity_list}

                    relevancy_scores[cargo] += sum(entity_count.values())

                    for weight in weight_type:

                        if weight in sentence:
                            entity_count = {str(weight): sentence.count(str(weight))}
                            relevancy_scores[cargo] += sum(entity_count.values())
                        else:
                            pass

                    for packaging in packaging_type:

                        if packaging in sentence:
                            entity_count = {str(packaging): sentence.count(str(packaging))}
                            relevancy_scores[cargo] += sum(entity_count.values())
                        else:
                            pass

                    else:
                        pass
                        #print(f"No relevant information found in {sentence}")

    for cargo, score in relevancy_scores.items():
        if score == 0:
            if isinstance(entities["Cargo"], list):
                if cargo in entities["Cargo"]:
                    entities["Cargo"].remove(cargo)
                    entities["Cargo"].append(f'{cargo} (might be irrelevant)')

    return entities


def remove_singular_quantities(entities):
    """
    Remove entities with keys starting with 'quantity_{packaging}' if the corresponding word is singular.

    Args:
        entities (dict): The dictionary containing entities.

    Returns:
        dict: The modified dictionary with singular quantity entities removed.
    """
    keys_to_remove = []

    for key in entities.keys():
        if key.startswith('quantity_'):
            word = key[len('quantity_'):]
            if is_singular(word):
                keys_to_remove.append(key)
    for key in keys_to_remove:
        del entities[key]
    return entities

def remove_plural_weights(entities):
    """
    Remove 'weight_per_{packaging}': 'information missing' if keys starting with 'weight_per_{packaging}' are plural.

    Args:
        entities (dict): The dictionary containing entities.

    Returns:
        dict: The modified dictionary with 'weight_per_{packaging}' removed if keys are plural.
    """
    if 'Weight' in entities:
        weight_info = entities['Weight']
        if isinstance(weight_info, dict):
            keys_to_remove = []
            for key in weight_info.keys():
                if key.startswith('weight_per_'):
                    word = key[len('weight_per_'):]
                    if not is_singular(word):
                        if len(entities['Weight']) != 1:
                            keys_to_remove.append(key)
            for key in keys_to_remove:
                del weight_info[key]
    return entities

def to_singular(word):
    """
    Convert a plural word to its singular form using NLTK's WordNet.

    Args:
        word (str): The word to convert.

    Returns:
        str: The singular form of the word.
    """

    if is_singular(word):
        return word

    # Exceptions for specific words
    exceptions = {
        'pallets': 'pallet',
        'Pallets': 'Pallet',
        'pallet': 'pallet',
        'Pallet': 'Pallet',
        'canisters': 'canister',
        'Canisters': 'Canister',
        'canister': 'canister',
        'Canister': 'Canister',
        'packs': 'pack',
        'Packs': 'Pack',
        'pack': 'pack',
        'Pack': 'Pack',
        'vials': 'vial',
        'Vials': 'Vial',
        'vial': 'vial',
        'Vial': 'Vial',
        'tubs': 'tub',
        'Tubs': 'Tub',
        'tub': 'tub',
        'Tub': 'Tub',
        'totes': 'tote',
        'Totes': 'Tote',
        'tote': 'tote',
        'Tote': 'Tote',
        'packets': 'packet',
        'Packets': 'Packet',
        'packet': 'packet',
        'Packet': 'Packet',
    }

    if word in exceptions:
        return exceptions[word]

    synsets = wordnet.synsets(word)
    if synsets:
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    return lemma.name()
    return word

def add_missing_weights(entities):

    """
    Add missing weight information for each packaging type to the 'Weight' dictionary.

    Args:
        entities (dict): The dictionary containing entities.

    Returns:
        dict: The modified dictionary with missing weight information added.
    """
    if 'Packaging' in entities and 'Weight' in entities:
        packaging_types = entities['Packaging']
        weight_info = entities['Weight']
        for packaging in packaging_types:
            singular_form = to_singular(packaging)
            weight_key = f'weight_per_{singular_form}'
            if weight_key not in weight_info:
                weight_info[weight_key] = 'information missing'
    return entities


def clean_entities_based_on_text(input_text, entities):
    sentences = re.split(r'(?<=[.!?]) +', input_text)
    keys_to_delete = []

    for key, value in entities.items():
        if '_contain' in key:
            packaging = key.split('_contain')[0]
            content_present = False

            for sentence in sentences:

                packaging_positions = [m.start() for m in re.finditer(packaging, sentence)]
                content_position = sentence.find(str(value))

                if len(packaging_positions) > 1:

                    if content_position < packaging_positions[0] and content_position != -1:
                        content_present = True
                elif len(packaging_positions) == 1:
                    if packaging in sentence and str(value) in sentence:

                        if abs(content_position - packaging_positions[0]) < len(sentence):
                            content_present = True

            if not content_present:
                keys_to_delete.append(key)


    for key in keys_to_delete:
        del entities[key]

    return entities
