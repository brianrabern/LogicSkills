from Database.DB import db
from Database.models import Argument, Sentence
import json
from Utils.normalize import unescape_logical_form
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

session = db.session

symbol_map = {"¬": "~3", "∧": "~1", "∨": "~2", "→": "~5", "↔": "~4", "∀": "~6", "∃": "~7"}


def get_argument(argument):
    print("Parsing argument: ", argument.id)
    language = argument.language
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=language).first()
    domain_constraint_form = unescape_logical_form(domain_constraint.form)

    premise_id_string = argument.premise_ids
    premise_ids = premise_id_string.split(",")
    conclusion_id = argument.conclusion_id

    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    premises_forms = [unescape_logical_form(premise.form) for premise in premises]

    conclusion = session.get(Sentence, conclusion_id)
    conclusion_form = unescape_logical_form(conclusion.form)

    argument_form = domain_constraint_form
    for premise_form in premises_forms:
        argument_form += "," + premise_form
    argument_form += "|=" + conclusion_form
    print(argument_form)

    # replace symbols with squggle encoiding
    argument_form = argument_form.replace("¬", "~3")
    argument_form = argument_form.replace("∧", "~1")
    argument_form = argument_form.replace("∨", "~2")
    argument_form = argument_form.replace("→", "~5")
    argument_form = argument_form.replace("↔", "~4")
    argument_form = argument_form.replace("∀", "~6")
    argument_form = argument_form.replace("∃", "~7")
    return argument_form


def parse_model_table(html):
    """Parse the model table HTML into a Python dictionary."""
    soup = BeautifulSoup(html, "html.parser")
    model = {}

    # Get all rows except the header
    rows = soup.find_all("tr")
    for row in rows:
        # Get the key (first column) and value (second column)
        cols = row.find_all("td")
        if len(cols) == 2:
            key = cols[0].text.strip().rstrip(":")  # Remove trailing colon
            value = cols[1].text.strip()

            # Parse the value based on its format
            if value.startswith("{") and value.endswith("}"):
                # Handle set notation
                if value == "{  }":  # Empty set
                    value = []
                else:
                    # Remove braces and split by comma
                    content = value[1:-1].strip()
                    if content:
                        # Handle tuples in the set
                        if "(" in content:
                            # Parse tuples like (1,0), (1,1)
                            # First split by closing parenthesis and comma
                            tuples = []
                            current = ""
                            for char in content:
                                if char == "(":
                                    current = "("
                                elif char == ")":
                                    current += ")"
                                    tuples.append(current)
                                    current = ""
                                elif current:
                                    current += char
                            value = [eval(t) for t in tuples]
                        else:
                            # Parse simple values like 0, 1, 2
                            value = [int(x.strip()) for x in content.split(",")]
            else:
                # Handle single values
                try:
                    value = int(value)
                except ValueError:
                    value = value  # Keep as string if not an integer

            model[key] = value

    return model


if __name__ == "__main__":
    problem_args = ["e89da97d6bb95d63"]  # "a70596728882c1b4",
    no_countermodel_found = []

    countermodels = {}

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the driver with automatic ChromeDriver management
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Load the base page once
        print("Loading base page...")
        driver.get("https://www.umsu.de/trees/")
        time.sleep(2)  # Give initial page time to load

        # Get all invalid arguments
        # invalid_arguments = session.query(Argument).filter_by(valid=False).all()
        invalid_arguments = session.query(Argument).filter(Argument.id.in_(problem_args)).all()
        # Process each argument
        for argument in invalid_arguments:
            print(f"Processing argument {argument.id}...")
            arg = get_argument(argument)

            # Update the URL with the new argument
            driver.get(f"https://www.umsu.de/trees/#{arg}")

            # Wait for either the model to appear or a timeout
            try:
                # Wait up to 10 seconds for the model div to be visible
                WebDriverWait(driver, 100).until(
                    lambda d: d.find_element(By.ID, "model").get_attribute("style") != "display: none;"
                )
            except Exception as e:
                print(f"Timeout waiting for model for argument {argument.id}: {e}")
                no_countermodel_found.append(argument.id)
                continue

            # Find the model div
            model_div = driver.find_element(By.ID, "model")
            model_div_style = model_div.get_attribute("style")
            if "display: none;" in model_div_style:
                print(f"No model found for argument {argument.id}")
                no_countermodel_found.append(argument.id)
                continue
            if model_div:
                # Get the model HTML
                model_html = model_div.get_attribute("outerHTML")

                # Parse the model into a dictionary
                model_dict = parse_model_table(model_html)
                print("Adding countermodel for argument: ", argument.id)
                countermodels[argument.id] = model_dict
            else:
                print(f"No model found for argument {argument.id}")
                no_countermodel_found.append(argument.id)

        with open("no_countermodel_found3.json", "w", encoding="utf-8") as f:
            json.dump(no_countermodel_found, f, indent=2)

        with open("countermodels3.json", "w", encoding="utf-8") as f:
            json.dump(countermodels, f, indent=2)

    finally:
        # Close the driver when done
        driver.quit()
