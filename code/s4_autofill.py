#!/usr/bin/env python3
"""
Universal Autofill Script with full résumé parsing without single-line comprehensions
Extracts contact info, skills, education, projects, experience, leadership from your résumé PDF
and autofills any application form via Selenium + ML heuristics.

Isn't 100% functional as 
"""

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from tensorflow.keras.layers import TextVectorization
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
import tensorflow as tf
import numpy as np
import pickle
import time
import re








# —— Load ML model & components —— #
model = tf.keras.models.load_model('../models/field_classifier_model.h5')
with open('../models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('../models/vectorizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vectorizer = TextVectorization(
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=20
)
vectorizer.set_vocabulary(vocab)

# —— Prompt for résumé path & extract text —— #
resume_path = input("Enter the absolute path to your résumé PDF: ").strip()
# Read all pages into a list of strings
page_texts = []
with pdfplumber.open(resume_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text is None:
            text = ""
        page_texts.append(text)
# Join pages into full_text
full_text = "\n".join(page_texts)

# —— Parse header: Name, Location, Email, Phone, LinkedIn —— #
# Build lines list without list comprehension
lines = []
for raw_line in full_text.splitlines():
    stripped = raw_line.strip()
    if stripped:
        lines.append(stripped)

full_name = ""
if len(lines) > 0:
    full_name = lines[0]

location = ""
email    = ""
phone    = ""
linkedin = ""

if len(lines) > 1:
    header_parts = lines[1].split('|')
    # Clean and assign
    if len(header_parts) > 0:
        location = header_parts[0].strip()
    if len(header_parts) > 1:
        email = header_parts[1].strip()
    if len(header_parts) > 2:
        phone = header_parts[2].strip()
    if len(header_parts) > 3:
        linkedin = header_parts[3].strip()
# Normalize LinkedIn
if linkedin and not linkedin.startswith('http'):
    linkedin = 'https://' + linkedin
# Split city and state
city = ''
state = ''
if location:
    parts = location.split(',', 1)
    city = parts[0].strip()
    if len(parts) > 1:
        state = parts[1].strip()

# —— Section parsing for full résumé —— #
raw_lines = full_text.splitlines()
section_titles = set([
    "SKILLS",
    "EDUCATION",
    "PROJECTS",
    "TECHNICAL EXPERIENCE",
    "WORK EXPERIENCE",
    "LEADERSHIP AND ENGAGEMENT"
])
sections = {}
current_section = None
for raw_line in raw_lines:
    up = raw_line.strip().upper()
    if up in section_titles:
        current_section = up
        sections[current_section] = []
    elif current_section:
        if raw_line.strip():
            sections[current_section].append(raw_line.strip())
# Convert lists to strings without comprehensions
skills_lines = []
education_lines = []
projects_lines = []
tech_lines = []
work_lines = []
lead_lines = []
if 'SKILLS' in sections:
    for item in sections['SKILLS']:
        skills_lines.append(item)
if 'EDUCATION' in sections:
    for item in sections['EDUCATION']:
        education_lines.append(item)
if 'PROJECTS' in sections:
    for item in sections['PROJECTS']:
        projects_lines.append(item)
if 'TECHNICAL EXPERIENCE' in sections:
    for item in sections['TECHNICAL EXPERIENCE']:
        tech_lines.append(item)
if 'WORK EXPERIENCE' in sections:
    for item in sections['WORK EXPERIENCE']:
        work_lines.append(item)
if 'LEADERSHIP AND ENGAGEMENT' in sections:
    for item in sections['LEADERSHIP AND ENGAGEMENT']:
        lead_lines.append(item)
# Join lines into text
skills_text = ' ; '.join(skills_lines)
education_text = ' | '.join(education_lines)
projects_text = ' | '.join(projects_lines)
tech_text = ' | '.join(tech_lines)
work_text = ' | '.join(work_lines)
lead_text = ' | '.join(lead_lines)

# —— Build personal_info dict —— #
personal_info = {
    "full_name":            full_name,
    "first_name":           full_name.split()[0] if full_name else "",
    "last_name":            full_name.split()[-1] if full_name else "",
    "email":                email,
    "phone_number":         phone,
    "address":              "",
    "city":                 city,
    "state":                state,
    "zipcode":              "",
    "country":              "",
    "linkedin":             linkedin,
    "portfolio_link":       linkedin,
    "linkedin_profile":     linkedin,
    "resume_upload":        resume_path,
    "skills":               skills_text,
    "education":            education_text,
    "projects":             projects_text,
    "technical_experience": tech_text,
    "work_experience":      work_text,
    "leadership":           lead_text
}

# —— Heuristic override before ML —— #
def heuristic_label(elem):
    typ = (elem.get_attribute('type') or '').lower()
    nm  = (elem.get_attribute('name') or '').lower()
    ph  = (elem.get_attribute('placeholder') or '').lower()
    label_attr = (elem.get_attribute('aria-label') or '').lower()
    # input types
    if typ == 'email':      return 'email'
    if typ == 'tel':        return 'phone_number'
    if typ == 'url':        return 'portfolio_link'
    if typ == 'file':       return 'resume_upload'
    # name/id hints
    for key in personal_info.keys():
        if key in ['skills','education','projects','technical_experience','work_experience','leadership']:
            continue
        token = key.replace('_','')
        if token in nm or token in label_attr or token in ph:
            return key
    # section hints
    if 'skill' in nm or 'skills' in ph:
        return 'skills'
    if 'educat' in nm or 'educat' in ph:
        return 'education'
    if 'project' in nm or 'project' in ph:
        return 'projects'
    if 'technical' in nm or 'technical' in ph:
        return 'technical_experience'
    if 'work' in nm or 'work' in ph:
        return 'work_experience'
    if 'leader' in nm or 'leader' in ph:
        return 'leadership'
    return None

# —— ML predictor fallback —— #
def predict_field(label_text):
    tensor = tf.convert_to_tensor([label_text])
    vect   = vectorizer(tensor)
    preds  = model.predict(vect)
    idx    = np.argmax(preds, axis=1)
    return label_encoder.inverse_transform(idx)[0]

# —— Robust click + type helper —— #
def click_and_type(elem, text, driver):
    try:
        elem.click()
    except:
        try:
            driver.execute_script('arguments[0].click();', elem)
        except:
            ActionChains(driver).move_to_element(elem).click().perform()
    time.sleep(0.5)
    elem.clear()
    elem.send_keys(text)
    time.sleep(0.5)

# —— Main Autofill Routine —— #
def main():
    url = input("Enter the URL to autofill: ").strip()
    options = Options()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'input')))

    inputs = driver.find_elements(By.TAG_NAME, 'input')
    for elem in inputs:
        try:
            # skip non-interactable
            if not elem.is_displayed() or not elem.is_enabled():
                continue
            typ = (elem.get_attribute('type') or '').lower()
            if typ not in {'text','email','tel','url','number','search','password','file'}:
                continue

            # —— LABEL DETECTION —— #
            label_text = ''
            try:
                lbl = elem.find_element(By.XPATH, 'ancestor::label')
                label_text = lbl.text.strip()
            except:
                pass
            if not label_text:
                iid = elem.get_attribute('id')
                if iid:
                    try:
                        lbl = driver.find_element(By.CSS_SELECTOR, f"label[for='{iid}']")
                        label_text = lbl.text.strip()
                    except:
                        pass
            if not label_text:
                try:
                    lbl = elem.find_element(By.XPATH, 'preceding-sibling::label[1]')
                    label_text = lbl.text.strip()
                except:
                    pass
            if not label_text:
                try:
                    lbl = elem.find_element(By.XPATH, 'parent::div/preceding-sibling::label[1]')
                    label_text = lbl.text.strip()
                except:
                    pass
            if not label_text:
                label_text = (elem.get_attribute('placeholder') or elem.get_attribute('aria-label') or '')
            if not label_text:
                continue
            # —— END LABEL DETECTION —— #

            key = heuristic_label(elem) or predict_field(label_text)
            print(f"Field: '{label_text}' → Key: {key}")
            if key not in personal_info:
                continue
            val = personal_info[key]

            # fill
            if typ == 'file':
                elem.send_keys(val)
            elif key == 'city':
                click_and_type(elem, val, driver)
                elem.send_keys(Keys.ARROW_DOWN)
                elem.send_keys(Keys.ENTER)
            else:
                click_and_type(elem, val, driver)

        except Exception as e:
            print(f"Skipping field due to error: {e}")
            continue
    print("✅ Autofill complete.")

if __name__ == '__main__':
    main()
