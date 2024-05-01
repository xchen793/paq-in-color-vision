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
    # Call the external Python script to get RGB directions and starting color
    data = get_color.getRandomColor()
    return render_template('test_ui.html', data = data)

@app.route('/<page_name>')
def render_page(page_name):
    try:
        return render_template(f'{page_name}.html')
    except Exception as e:
        return f"Error loading the page: {str(e)}", 404


@app.route('/submit-color', methods=['POST'])
def submit_color():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Extracting individual components from the data
        xyData = data.get('xyData')
        fixedColor = data.get('fixedColor')
        pageId = data.get('pageId')  # This should be sent from the frontend

        # File path for JSON storage
        file_path = 'color_data.json'
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                try:
                    existing_data = json.load(file)
                    existing_data[pageId] = {
                        'xyData': xyData,
                        'fixedColor': fixedColor
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
                        'xyData': xyData,
                        'fixedColor': fixedColor
                    }
                }, file, indent=4)

        return jsonify(status="Success", message="Data stored successfully")

    except Exception as e:
        return jsonify(status='Error', message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)