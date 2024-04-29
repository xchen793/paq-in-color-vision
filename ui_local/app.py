from flask import Flask, render_template, jsonify
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

if __name__ == '__main__':
    app.run(debug=True)
