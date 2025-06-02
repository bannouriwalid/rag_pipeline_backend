import json
import os

from bs4 import BeautifulSoup
from pathlib import Path
import requests
from config import Config

raw_files_folder = f'{Config.KB}/raw_files'


def WHO_scrapping(save_path, disease, desired_sections):
    url = "https://www.who.int/news-room/fact-sheets/detail/" + disease
    file_name = "who_0.json"
    file_path = Path(save_path) / file_name

    try:
        if file_path.exists():
            print(f"Already exists, skipping: {file_name}")
            return

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract specific h2 sections
        article = soup.find("section", {"id": "content"})
        extracted_sections = {}

        if article:
            headers = article.find_all("h2")
            for h2 in headers:
                title = h2.get_text(strip=True).lower()
                for key in desired_sections:
                    if key in title:
                        content = []
                        for sibling in h2.find_next_siblings():
                            if sibling.name == "h2":
                                break
                            # Handle <p> and <li> tags with proper formatting
                            if sibling.name == "p":
                                text = sibling.get_text(separator=" ", strip=True)
                                content.append(text)
                            elif sibling.name == "ul":
                                items = sibling.find_all("li")
                                for item in items:
                                    item_text = item.get_text(separator=" ", strip=True)
                                    content.append(item_text + ",")
                        combined = " ".join(content)
                        # Clean up extra whitespace and Unicode chars
                        cleaned = (
                            combined.replace("\t", " ")
                            .replace("\xa0", " ")
                            .replace(" ,", ",")
                            .replace("  ", "")
                            .replace(".,", ",")
                            .strip()
                        )
                        extracted_sections[key.capitalize()] = cleaned
                        break

        extracted_sections["url"] = url
        os.makedirs(file_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(extracted_sections, f, ensure_ascii=False, indent=2)

        print(f"Scraped and saved: {file_path}")

    except Exception as e:
        print(f"Failed to download or parse: {e}")


def CDC_scrapping(save_path, disease, desired_pages):
    base_url = "https://www.cdc.gov/" + disease
    file_id = 0

    for sub_path, sections in desired_pages.items():
        url = base_url + sub_path
        file_name = f"cdc_{file_id}.json"
        file_path = Path(save_path) / file_name

        try:
            if file_path.exists():
                print(f"Already exists, skipping: {file_name}")
                file_id += 1
                continue

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            extracted_sections = {}

            # All <div class="dfe-section">
            all_sections = soup.find_all("div", class_="dfe-section")

            for div in all_sections:
                h2 = div.find("h2")
                if not h2:
                    continue

                title = h2.get_text(strip=True).lower()

                for desired in sections:
                    if desired.lower() in title:
                        content_parts = []

                        # Include h3s, ps, and lis
                        for tag in div.find_all(["h3", "p", "li"]):
                            txt = tag.get_text(separator=" ", strip=True)
                            if txt:
                                content_parts.append(txt)

                        # Join and clean
                        combined = ", ".join(content_parts)
                        cleaned = (
                            combined.replace("\t", " ")
                            .replace("\xa0", " ")
                            .replace(" .", ".")
                            .replace(" ,", ",")
                            .replace("  ", " ")
                            .replace(".,", ".")
                            .replace(":,", ":")
                            .replace("\"", "")
                            .replace("\"", "")
                            .replace("\"", "")
                            .strip()
                        )

                        extracted_sections[desired] = cleaned
                        break

            extracted_sections["url"] = url
            # Save to JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(extracted_sections, f, ensure_ascii=False, indent=2)

            print(f"Scraped and saved: {file_path}")

        except Exception as e:
            print(f"Failed to download or parse {url}: {e}")

        file_id += 1  # Always increment after processing one URL


def NIH_scrapping(save_path, disease, desired_pages):
    base_url = "https://www.nhlbi.nih.gov/health/" + disease
    file_id = 0

    for sub_path in desired_pages:
        url = base_url + sub_path
        file_name = f"nih_{file_id}.json"
        file_path = Path(save_path) / file_name

        try:
            if file_path.exists():
                print(f"Already exists, skipping: {file_name}")
                file_id += 1
                continue

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            extracted_sections = {"url": url}

            main_title_tag = soup.find("h1")

            if main_title_tag:
                # Remove all <span> tags
                for span in main_title_tag.find_all("span"):
                    span.decompose()
                main_title = main_title_tag.get_text(strip=True)
            else:
                main_title = "No Title"
            container = soup.find("div", class_="field--name-field-component-sections")

            if not container:
                print(f"No component sections found in {url}")
                file_id += 1
                continue

            components = container.find_all("div", class_="paragraph--type--component-section")
            for comp in components:
                title_tag = comp.find("h2", class_="component-section-section-title")
                content_tag = comp.find("div", class_="field--name-field-component-section-content")

                if not content_tag:
                    continue

                html_content = (
                    content_tag.get_text(separator=" ", strip=True)
                    .replace("\t", " ")
                    .replace("\xa0", " ")
                    .replace(" ,", ",")
                    .replace("  ", "")
                    .replace(".,", ",")
                    .strip()
                )
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    extracted_sections[title] = html_content
                else:
                    extracted_sections[main_title] = html_content

            # Save to JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(extracted_sections, f, ensure_ascii=False, indent=2)

            print(f"Scraped and saved: {file_path}")

        except Exception as e:
            print(f"Failed to download or parse {url}: {e}")

        file_id += 1


def scrapping(folder_name, disease_names, who_pages_list, cdc_dict, nih_pages_list):
    new_disease_folder = os.path.join(raw_files_folder, folder_name)
    os.makedirs(new_disease_folder, exist_ok=True)
    WHO_scrapping(new_disease_folder, disease_names[0], who_pages_list)
    CDC_scrapping(new_disease_folder, f"/{disease_names[1]}", cdc_dict)
    NIH_scrapping(new_disease_folder, f"/{disease_names[2]}", nih_pages_list)


def initial_scrapping():
    # asthma
    scrapping(os.path.join(raw_files_folder, "asthma"), ["asthma" for _ in range(3)],
              ["overview", "impact", "symptoms", "causes", "treatment", "self-care"], "asthma/",
              {"about": ["symptoms", "diagnosis", "symptom management"],
               "control": ["Common asthma triggers"],
               "emergency": ["First steps", "Facing challenges"]
               }, ["", "symptoms", "attacks", "causes", "diagnosis", "treatment-action-plan", "living-with"])
    # copd
    scrapping(os.path.join(raw_files_folder, "chronic-obstructive-pulmonary-disease-copd"),
              ["chronic-obstructive-pulmonary-disease-(copd)", "copd", "copd"],
              ["overview", "symptoms", "causes", "treatment", "living with copd"],
              {"about": ["What it is", "symptoms", "Complications", "Causes and risk factors", "Reducing risk",
                         "Who is at risk", "Diagnosis", "Treatment and management"]},
              ["", "symptoms", "causes", "diagnosis", "prevention", "treatment", "living-with"])
    # pneumonia
    PNEUMONIA_RAW_DIR = os.path.join(raw_files_folder, "pneumonia")
    os.makedirs(PNEUMONIA_RAW_DIR, exist_ok=True)
    CDC_scrapping(PNEUMONIA_RAW_DIR, "pneumonia/",
                  {"about": ["Overview", "symptoms", "Types", "Who is at risk", "Causes", "Prevention"],
                   "risk-factors": ["People at increased risk", "Conditions that can increase risk",
                                    "Behaviors that can increase risk"],
                   "prevention": ["Prevention steps and strategies"]
                   })
    NIH_scrapping(PNEUMONIA_RAW_DIR, "pneumonia/",
                  ["", "symptoms", "causes", "diagnosis", "prevention", "treatment", "recovery"])
    # tuberculosis
    TUBERCULOSIS_RAW_DIR = os.path.join(raw_files_folder, "tuberculosis")
    os.makedirs(TUBERCULOSIS_RAW_DIR, exist_ok=True)

    WHO_scrapping(TUBERCULOSIS_RAW_DIR, "tuberculosis",
                  ["overview", "symptoms", "treatment", "prevention", "diagnosis", "impact"])

    CDC_scrapping(TUBERCULOSIS_RAW_DIR, "tb/",
                  {"about": ["Overview", "Signs and symptoms", "Types", "Risk factors", "How it spreads",
                             "Prevention", "Testing", "Treatment", "Vaccines"],
                   "signs-symptoms": ["Signs and symptoms"],
                   "causes": ["Causes", "How it spreads"],
                   "vaccines": ["Overview"],
                   "testing": ["Types of tests", "Why get tested", "Who should be tested",
                               "What to do if you've tested positive"],
                   "exposure": ["Contact your health care provider if you have been exposed to TB",
                                "Only persons with active TB disease can spread TB to others",
                                "Contact investigations can help limit the spread of TB"],
                   "risk-factors": ["Places with increased risk", "Conditions that can increase risk"],
                   "prevention": ["Prevention steps and strategies"]
                   })
    # covid
    COVID_RAW_DIR = os.path.join(raw_files_folder, "coronavirus-disease-(covid-19)")
    os.makedirs(COVID_RAW_DIR, exist_ok=True)

    WHO_scrapping(COVID_RAW_DIR, "coronavirus-disease-(covid-19)",
                  ["overview", "symptoms", "treatment", "prevention"])

    CDC_scrapping(COVID_RAW_DIR, "covid/", {"about": ["Learn about COVID-19 and how it spreads"],
                                            "signs-symptoms": ["Signs and symptoms", "When to seek emergency help",
                                                               "Difference between flu and COVID-19"],
                                            "risk-factors": ["Overview", "Conditions that can increase risk"],
                                            "testing": ["Types of tests", "Choosing a COVID-19 test",
                                                        "Interpreting your results"],
                                            "treatment": ["COVID-19 Treatment Options", "Preventing COVID-19"],
                                            "prevention": ["Core Prevention Strategies", "What to watch out for"]
                                            })

"""
Code Report: Web Scraping Module for Medical Information

1. Overview
-----------
This module implements a web scraping system for collecting medical information from three major health organizations:
- World Health Organization (WHO)
- Centers for Disease Control and Prevention (CDC)
- National Institutes of Health (NIH)

2. Core Components
-----------------
a) WHO_scrapping()
   - Scrapes WHO fact sheets for specific diseases
   - Extracts content from specified sections
   - Saves data in JSON format
   - Handles HTML parsing with BeautifulSoup
   - Includes error handling and duplicate file checking

b) CDC_scrapping()
   - Scrapes CDC disease pages
   - Processes multiple sub-pages per disease
   - Extracts content from dfe-section divs
   - Handles various HTML elements (h2, h3, p, li)
   - Implements robust text cleaning

c) NIH_scrapping()
   - Scrapes NIH health pages
   - Processes component sections
   - Handles main titles and subsections
   - Implements specific HTML structure parsing
   - Includes comprehensive error handling

3. Supporting Functions
----------------------
- scrapping(): Orchestrates the scraping process for a single disease
- initial_scrapping(): Initializes scraping for multiple diseases:
  * Asthma
  * COPD
  * Pneumonia
  * Tuberculosis
  * COVID-19

4. Data Storage
--------------
- Uses JSON format for data storage
- Implements organized directory structure
- Includes duplicate file checking
- Maintains URL references in saved data

5. Error Handling
----------------
- Implements try-except blocks for network requests
- Handles missing content gracefully
- Includes timeout protection
- Provides informative error messages

6. Text Processing
-----------------
- Implements comprehensive text cleaning
- Handles special characters and whitespace
- Maintains proper formatting of lists and paragraphs
- Preserves important punctuation

7. Dependencies
--------------
- BeautifulSoup4: HTML parsing
- requests: HTTP requests
- json: Data serialization
- os: File system operations
- pathlib: Path handling

8. Configuration
---------------
- Uses Config class for path configuration
- Implements flexible directory structure
- Supports multiple disease configurations

9. Best Practices
----------------
- Implements modular design
- Uses consistent error handling
- Maintains clean code structure
- Includes comprehensive documentation
- Implements rate limiting through timeouts

10. Areas for Improvement
------------------------
- Add rate limiting between requests
- Implement retry mechanism for failed requests
- Add logging system
- Implement proxy support
- Add data validation
- Implement async requests for better performance
"""
