"""
This is a weird script that uses Wolfgang's Tree Proof Generator to get countermodels for invalid arguments.
It's a bit of a hack, but it works. Wo's algorithm finds the minimal countermodels, so are nicer than the often large ones found by the Z3 solver.
https://github.com/wo/tpg
"""

import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
from bs4 import BeautifulSoup

symbol_map = {"¬": "~3", "∧": "~1", "∨": "~2", "→": "~5", "↔": "~4", "∀": "~6", "∃": "~7"}


def get_argument_from_question(question: str) -> str:
    """Encode a logical argument string for TPG by replacing unicode symbols with squiggle codes.

    The input is expected to look like: "φ1, φ2, ..., |= ψ".
    """
    argument_form = question
    # replace symbols with squiggle encoding
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

    # get all rows except the header
    rows = soup.find_all("tr")
    for row in rows:
        # get the key (first column) and value (second column)
        cols = row.find_all("td")
        if len(cols) == 2:
            key = cols[0].text.strip().rstrip(":")
            value = cols[1].text.strip()

            # parse the value based on its format
            if value.startswith("{") and value.endswith("}"):
                # handle set notation
                if value == "{  }":  # empty set
                    value = []
                else:
                    # remove braces and split by comma
                    content = value[1:-1].strip()
                    if content:
                        # handle tuples in the set
                        if "(" in content:
                            # Pparse tuples like (1,0), (1,1)
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
                            # parse simple values like 0, 1, 2
                            value = [int(x.strip()) for x in content.split(",")]
            else:
                # handle single values
                try:
                    value = int(value)
                except ValueError:
                    value = value  # keep as string if not an integer

            model[key] = value

    return model


if __name__ == "__main__":
    no_countermodel_found = []
    countermodels = {}

    # set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # initialize the driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # load the base page once
        print("Loading base page...")
        driver.get("https://www.umsu.de/trees/")
        time.sleep(2)  # give initial page time to load

        # get all invalid arguments from the local JSON file produced by the extractor
        with open(
            "Assessors/countermodel/extras/extracted_questions_countermodel_prelim.json", "r", encoding="utf-8"
        ) as f:
            invalid_arguments = json.load(f)

        # process each argument
        for item in invalid_arguments:
            arg_id = item["id"]
            print(f"Processing argument {arg_id}...")
            arg = get_argument_from_question(item["question"])

            # update the URL with the new argument
            driver.get(f"https://www.umsu.de/trees/#{arg}")

            # wait for either the model to appear or a timeout
            try:
                # wait up to 100 seconds for the model div to be visible
                WebDriverWait(driver, 100).until(
                    lambda d: d.find_element(By.ID, "model").get_attribute("style") != "display: none;"
                )
            except Exception as e:
                print(f"Timeout waiting for model for argument {arg_id}: {e}")
                no_countermodel_found.append(arg_id)
                continue

            # find the model div
            model_div = driver.find_element(By.ID, "model")
            model_div_style = model_div.get_attribute("style")
            if "display: none;" in model_div_style:
                print(f"No model found for argument {arg_id}")
                no_countermodel_found.append(arg_id)
                continue
            if model_div:
                # get the model HTML
                model_html = model_div.get_attribute("outerHTML")

                # parse the model into a dictionary
                model_dict = parse_model_table(model_html)
                print("Adding countermodel for argument: ", arg_id)
                countermodels[arg_id] = model_dict
            else:
                print(f"No model found for argument {arg_id}")
                no_countermodel_found.append(arg_id)

        with open("no_countermodel_found4.json", "w", encoding="utf-8") as f:
            json.dump(no_countermodel_found, f, indent=2)

        with open("countermodels4.json", "w", encoding="utf-8") as f:
            json.dump(countermodels, f, indent=2)

    finally:
        # close the driver
        driver.quit()
