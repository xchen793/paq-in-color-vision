"""
===============================================================================
Filename: app.py
Description: Python file for web application deployment

Author: Xuanzhou Chen
Date Created: August 21, 2024

Usage:
    Bash:
        $ python app.py 
    Copy and paste the local IP address to run the app.

Notes:
    - Include both main experiment and model validation.

===============================================================================
"""

from flask import Flask, request, render_template, jsonify
import json
import os
import traceback
import fcntl


app = Flask(__name__)


DATA_FILE = 'data/color_data.json' # when run locally, uncomment this and comment the liene below
MODEL_FILE = 'model_selection/data/model_rejection.json'

def read_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def write_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

def read_data_file(FILE):
    if os.path.exists(FILE):
        try:
            with open(FILE, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print("JSON decode error while reading:", str(e))
            return {}
        except Exception as e:
            print("Error reading the JSON file:", str(e))
            return {}
    return {}

def write_data_file(data, FILE):
    try:
        with open(FILE, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print("Error writing to the JSON file:", str(e))
        raise

@app.route('/')
def index():
    return render_template('prolific_id.html')

@app.route('/consent')
def consent():
    return render_template('consent.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/failure')
def failure():
    return render_template('failure.html')

@app.route('/pre_survey')
def pre_survey():
    return render_template('pre_survey.html')

#######   model rejection page + data handler   #######  
# Route for model_rejection.html
@app.route('/model_rejection')
def model_rejection():
    return render_template('model_rejection.html')

# Route for questions.html
@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/submit_model_rejection', methods=['POST'])
def submit_model_rejection():
    try:
        data = read_data_file(MODEL_FILE)
        new_entry = request.json

        if not new_entry:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        if 'pageId' not in new_entry:
            return jsonify({'status': 'error', 'message': 'pageId is missing'}), 400

        page_key = new_entry.pop('pageId', f"survey_page{len(data.get('model_rejection', {})) + 1}")
        if 'model_rejection' not in data:
            data['model_rejection'] = {}

        data['model_rejection'][page_key] = new_entry
        write_data_file(data, MODEL_FILE)

        return jsonify({'status': 'success'})

    except json.JSONDecodeError as e:
        print("JSON decode error:", str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
    except Exception as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


######### #############

@app.route('/survey')
def survey():
    page_number = request.args.get('page', default=1, type=int)
    return render_template('test_ui.html', page_number=page_number)

# # store the questions data
# @app.route('/api/survey', methods=['POST'])
# def save_survey():
#     data = read_data()
#     new_entry = request.get_json()
#     data.append({'Questions': new_entry})
#     write_data(data)
#     return jsonify({'message': 'Survey results saved successfully'}), 201

# store the color slider data
@app.route('/submit-color', methods=['POST'])
def submit_color():
    try:
        data = request.get_json()

        prolific_id = data.get('prolific_id')
        fixedColor = data.get('fixedColor')
        pageId = data.get('pageId')  
        query_vec = data.get('query_vec')  
        gamma = data.get('gamma')  
        endColor = data.get('endColor') 
        timeTaken = data.get('timeTaken') 


        if not prolific_id or not prolific_id.strip():
            return jsonify({"error": "Prolific ID is required"}), 400
        
        stored_data = load_data()

        if prolific_id not in stored_data:
            stored_data[prolific_id] = {}

        stored_data[prolific_id][pageId] = {
            'fixedColor': fixedColor,
            'query_vec': query_vec,
            'gamma': gamma,
            'endColor': endColor,
            'timeTaken': timeTaken
        }

        save_data(stored_data)

        return jsonify(status="Success", message="Data stored successfully")

    except Exception as e:
        return jsonify(status='Error', message=str(e)), 500

@app.route('/self_report')
def self_report():
    return render_template('self_report.html')

@app.route('/comments')
def comment():
    return render_template('comments.html')

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')


def load_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump({}, f)

    with open(DATA_FILE, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH) 
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)  
    return data

def save_data(data):
    with open(DATA_FILE, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX) 
        json.dump(data, f, indent=4)
        f.truncate()  
        fcntl.flock(f, fcntl.LOCK_UN)  

@app.route('/store_prolific_id', methods=['POST'])
def store_prolific_id():
    data = request.get_json()
    prolific_id = data.get('prolific_id')

    if not prolific_id:
        return jsonify({"error": "Prolific ID is required"}), 400

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as file:
            json.dump({}, file)

    stored_data = load_data()
    if prolific_id in stored_data:
        return jsonify({"error": "This Prolific ID has already been recorded. Please do not submit again."}), 400

    stored_data[prolific_id] = {}
    save_data(stored_data)

    return jsonify({"status": "success", "prolific_id": prolific_id}), 201


@app.route('/save-answer', methods=['POST'])
def save_answer():
    data = request.get_json()
    prolific_id = data.get('prolificId')
    color_blind_info = data.get('color_blindness')
    add_info = data.get('add_info')

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as file:
            json.dump({}, file)

    stored_data = load_data()
    stored_data[prolific_id]["color-blind-info"] = color_blind_info
    stored_data[prolific_id]["add-info"] = add_info
    save_data(stored_data)

    return jsonify(status="Success", message="Data stored successfully")

@app.route('/save-comments', methods=['POST'])
def save_comments():
    data = request.get_json()
    prolific_id = data.get('prolificId')
    comments = data.get('comments')

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as file:
            json.dump({}, file)

    stored_data = load_data()
    stored_data[prolific_id]["comments"] = comments
    save_data(stored_data)

    return jsonify(status="Success", message="Data stored successfully")

if __name__ == '__main__':
    app.run(port=5000, debug=True)