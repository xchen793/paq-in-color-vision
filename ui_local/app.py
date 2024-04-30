from flask import Flask, render_template, request, jsonify
from jinja2 import Template
import get_color
import subprocess
import json


app = Flask(__name__)

@app.route('/')
def index():
    # Call the external Python script to get RGB directions and starting color
    data = get_color.getRandomColor()
    return render_template('test_ui.html', data = data)

@app.route('/update_slider', methods=['POST'])
def update_slider():
    # Get the slider value from the request
    slider_value = request.json.get('sliderValue')
    # Handle the slider value as needed
    print("Slider value:", slider_value)
    # You can perform any processing or database operations here
    
    # Return a response if needed
    return jsonify({"message": "Slider value received successfully"})

if __name__ == '__main__':
    app.run(debug=True)
