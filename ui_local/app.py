from flask import Flask, request, render_template, jsonify
from jinja2 import Template
import get_color
import subprocess
import json
import time
import os
import traceback


app = Flask(__name__)

# store the questions data which decides whether a user will be paid or not
DATA_FILE = 'color_data.json'
MODEL_FILE = 'model_rejection.json'

def read_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def write_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

def read_data_model_rej():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print("JSON decode error while reading:", str(e))
            return {}
        except Exception as e:
            print("Error reading the JSON file:", str(e))
            return {}
    return {}

def write_data_model_rej(data):
    try:
        with open(MODEL_FILE, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print("Error writing to the JSON file:", str(e))
        raise

@app.route('/')
def index():
    return render_template('intro.html')

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
        data = read_data_model_rej()
        new_entry = request.json

        if not new_entry:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        if 'pageId' not in new_entry:
            return jsonify({'status': 'error', 'message': 'pageId is missing'}), 400

        page_key = new_entry.pop('pageId', f"survey_page{len(data.get('model_rejection', {})) + 1}")
        if 'model_rejection' not in data:
            data['model_rejection'] = {}

        data['model_rejection'][page_key] = new_entry
        write_data_model_rej(data)

        return jsonify({'status': 'success'})

    except json.JSONDecodeError as e:
        print("JSON decode error:", str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
    except Exception as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500



# Endpoint to handle data from questions.html
@app.route('/submit_questions', methods=['POST'])
def submit_questions():
    data = read_data_model_rej()
    new_entry = request.get_json()
    data['model_rejection'] = new_entry
    write_data_model_rej(data)
    return jsonify({'message': 'Survey results saved successfully'}), 201


######### #############

@app.route('/self-report')
def self_report():
    return render_template('self_report.html')

@app.route('/pre_test')
def pre_test():
    return render_template('pre_test.html')


@app.route('/survey')
def survey():
    page_number = request.args.get('page', default=1, type=int)
    return render_template('test_ui.html', page_number=page_number)

# store the questions data
@app.route('/api/survey', methods=['POST'])
def save_survey():
    data = read_data()
    new_entry = request.get_json()
    data.append({'Questions': new_entry})
    write_data(data)
    return jsonify({'message': 'Survey results saved successfully'}), 201

# store the color slider data
@app.route('/submit-color', methods=['POST'])
def submit_color():
    try:
        data = request.get_json()

        # Extracting individual components from the data
        fixedColor = data.get('fixedColor')
        pageId = data.get('pageId')  
        query_vec = data.get('query_vec')  
        gamma = data.get('gamma')  
        endColor = data.get('endColor') 

        # File path for JSON storage
        file_path = MODEL_FILE
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                try:
                    existing_data = json.load(file)
                    existing_data[pageId] = {
                        'fixedColor': fixedColor,
                        'query_vec': query_vec,
                        'gamma': gamma,
                        'endColor': endColor
                    }
                    file.seek(0)
                    json.dump(existing_data, file, indent=4)
                    file.truncate()
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            with open(file_path, 'w') as file:
                json.dump({
                    pageId: {
                        'fixedColor': fixedColor,
                        'query_vec': query_vec,
                        'gamma': gamma,
                        'endColor': endColor
                    }
                }, file, indent=4)

        return jsonify(status="Success", message="Data stored successfully")

    except Exception as e:
        return jsonify(status='Error', message=str(e)), 500

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)