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
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
from bs4 import BeautifulSoup
import gc

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


def create_driver():
    """Create a new Chrome driver instance with optimized settings."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--max_old_space_size=4096")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def process_arguments_batch(invalid_arguments, start_idx=0, batch_size=100):
    """Process arguments in batches to avoid memory issues."""
    no_countermodel_found = []
    countermodels = {}

    # Process in batches
    for batch_start in range(start_idx, len(invalid_arguments), batch_size):
        batch_end = min(batch_start + batch_size, len(invalid_arguments))
        batch_args = invalid_arguments[batch_start:batch_end]

        print(f"Processing batch {batch_start//batch_size + 1}: arguments {batch_start+1}-{batch_end}")

        # Create fresh driver for each batch
        driver = create_driver()

        try:
            # Load base page
            driver.get("https://www.umsu.de/trees/")
            time.sleep(2)
            # load Assessors/countermodel/extras/no_countermodel_found_combined.json
            with open("Assessors/countermodel/extras/no_countermodel_found_combined.json", "r") as f:
                no_countermodel_found_combined = json.load(f)

            for i, item in enumerate(batch_args):
                if item["id"] not in no_countermodel_found_combined:
                    continue
                arg_id = item["id"]
                print(f"  Processing argument {batch_start + i + 1}/{len(invalid_arguments)} (ID: {arg_id})...")

                try:
                    arg = get_argument_from_question(item["question"])
                    driver.get(f"https://www.umsu.de/trees/#{arg}")

                    # Wait for model with shorter timeout
                    try:
                        WebDriverWait(driver, 60).until(
                            lambda d: d.find_element(By.ID, "model").get_attribute("style") != "display: none;"
                        )
                    except TimeoutException:
                        print(f"    Timeout waiting for model for argument {arg_id}")
                        no_countermodel_found.append(arg_id)
                        continue

                    # Check if model exists
                    model_div = driver.find_element(By.ID, "model")
                    model_div_style = model_div.get_attribute("style")
                    if "display: none;" in model_div_style:
                        print(f"    No model found for argument {arg_id}")
                        no_countermodel_found.append(arg_id)
                        continue

                    # Parse model
                    model_html = model_div.get_attribute("outerHTML")
                    model_dict = parse_model_table(model_html)
                    print(f"    ✓ Found countermodel for argument {arg_id}")
                    countermodels[arg_id] = model_dict

                except WebDriverException as e:
                    print(f"    WebDriver error for argument {arg_id}: {e}")
                    no_countermodel_found.append(arg_id)
                    continue
                except Exception as e:
                    print(f"    Unexpected error for argument {arg_id}: {e}")
                    no_countermodel_found.append(arg_id)
                    continue

        finally:
            # Clean up driver
            try:
                driver.quit()
            except:
                pass
            gc.collect()  # Force garbage collection

        # Save progress after each batch
        with open(
            f"Assessors/countermodel/extras/donkey_no_countermodel_found_batch_{batch_start//batch_size + 1}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(no_countermodel_found, f, indent=2)

        with open(
            f"Assessors/countermodel/extras/donkey_countermodels_batch_{batch_start//batch_size + 1}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(countermodels, f, indent=2)

        print(
            f"  Batch {batch_start//batch_size + 1} completed. Found {len(countermodels)} countermodels, {len(no_countermodel_found)} failed."
        )

    return no_countermodel_found, countermodels


if __name__ == "__main__":
    # Load arguments
    with open("Assessors/countermodel/extras/extracted_questions_100000.json", "r", encoding="utf-8") as f:
        invalid_arguments = json.load(f)

    print(f"Loaded {len(invalid_arguments)} arguments to process")

    # Process in batches of 50 (adjust as needed)
    # Starting from batch 1448 (batch_size=50, so start_idx = (1448-1) * 50 = 72350)
    start_idx = 0
    print(f"Resuming from batch 1448 (argument index {start_idx})")
    no_countermodel_found, countermodels = process_arguments_batch(
        invalid_arguments, start_idx=start_idx, batch_size=50
    )

    # Final save
    with open("Assessors/countermodel/extras/donkey_no_countermodel_found_final.json", "w", encoding="utf-8") as f:
        json.dump(no_countermodel_found, f, indent=2)

    with open("Assessors/countermodel/extras/donkey_countermodels_final.json", "w", encoding="utf-8") as f:
        json.dump(countermodels, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Total countermodels found: {len(countermodels)}")
    print(f"Total failed: {len(no_countermodel_found)}")
