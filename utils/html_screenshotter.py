from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import time
import argparse
import os
import json
from tqdm import tqdm
import random


def setup_driver(width, height):
    # Set up the Chrome WebDriver options for each function call
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # Initialize and return the WebDriver with the specified options
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def html_to_screenshot(html_path, screenshot_path, width, height):
    driver = setup_driver(width, height)
    logs = driver.get_log('browser')
    try:
        # Open the HTML file
        driver.get(f"file://{html_path}")

        html_source = driver.page_source
        # print(f"The HTML source is: {html_source}")

        # Give the page some time to render (if necessary)
        time.sleep(2)

        # Save screenshot
        driver.save_screenshot(screenshot_path)

        # Optionally, crop the image to the window size
        img = Image.open(screenshot_path)
        img = img.crop((0, 0, width, height))
        img.save(screenshot_path)
    finally:
        # Ensure the driver is closed to free up resources
        driver.quit()
    for entry in logs:
        print(entry)  # This can help identify JavaScript errors or resource loading issues


def create_temp_html(html_template, title, text):
    with open(html_template, 'r') as f:
        html = f.read()
    html = html.format(TITLE=title, TEXT=text)

    # get the direcgory of the html_template
    html_template_dir = os.path.dirname(html_template)
    temp_html_path = os.path.join(html_template_dir, 'temp.html')
    with open(temp_html_path, 'w') as f:
        f.write(html)
    return temp_html_path

def test_func():
    # Example usage
    html_path = '/home/dcml0714/Downloads/html5/txt/no-sidebar.html'  # Update with your HTML file's path
    # html_path = '/work3/rag-metadata/data/raw.html'  # Update with your HTML file's path
    screenshot_path = 'screenshot.png'
    width = 1920
    height = 1280

    html_to_screenshot(html_path, screenshot_path, width, height)

def main(args):
    if not os.path.exists(args.img_output_path):
        os.makedirs(args.img_output_path)
    
    # Fix all seeds
    random.seed(args.seed)

    # Load dataset
    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    
    pbar = tqdm(enumerate(data), total=len(data))
    for i, instance in pbar:
        '''
        data_format = {
            'search_query': search_query,
            'search_engine_input': [search_engine_input],
            'search_type': [search_type],
            'urls': [url],
            'titles': [title],
            'text_raw': [text_raw],
            'text_window': [text_window],
            'stance': [stance]
        }
        '''
        # question = instance['search_query']
        
        # Sample two documents with different stances
        yes_index = [j for j in range(len(instance['stance'])) if instance['stance'][j] == 'yes']
        no_index = [j for j in range(len(instance['stance'])) if instance['stance'][j] == 'no']
        if len(yes_index) == 0 or len(no_index) == 0:
            continue
        yes_index = random.choice(yes_index)
        no_index = random.choice(no_index)
        for doc_idx in [yes_index, no_index]:
            TITLE = instance['titles'][doc_idx]
            TEXT = instance['text_window'][doc_idx]
            stance = instance['stance'][doc_idx]
            
            for html_template_type in ["simple", "pretty", "photo"]:
                html_template = getattr(args, f"{html_template_type}_html")
                if html_template == "":
                    continue

                temp_html_path = create_temp_html(html_template, TITLE, TEXT)
                screenshot_path = os.path.join(args.img_output_path, f"{stance}_{html_template_type}_{i}.png")
                html_to_screenshot(temp_html_path, screenshot_path, args.width, args.height)
                os.remove(temp_html_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_html', type=str, default = "/work3/rag-metadata/data/raw.html")
    parser.add_argument('--pretty_html', type=str, default = "/home/dcml0714/Downloads/html5/txt/no-sidebar.html")
    parser.add_argument('--photo_html', type=str, default = "")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--img_output_path', type=str, default = "../data/imgs/fake")
    parser.add_argument('--width', type=int, default = 1920)
    parser.add_argument('--height', type=int, default = 1280)
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()
    main(args)

    '''
    Usage: 
    python3 html_screenshotter.py \
      --dataset_path  /work3/rag-metadata/data/fake_knowledge_with_evidence_parsed.json
    '''