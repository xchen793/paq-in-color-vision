from flask import Flask, request, render_template, jsonify
from jinja2 import Template
import get_color
import subprocess
import json
import time
import os


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('intro.html')

@app.route('/survey')
def survey():
    # You can use a query parameter to determine the page number or state
    page_number = request.args.get('page', default=1, type=int)
    # You could add logic here to handle different page numbers or states
    return render_template('test_ui.html', page_number=page_number)

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

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
        file_path = 'color_data.json'
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

if __name__ == '__main__':
    app.run(debug=True)