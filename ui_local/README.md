
# Local app
## Getting started
This was tested with python 3.11.5. 
- Create a new conda environment: `conda create --name paqui -c conda-forge python=3.11.5`
- Activate environment: `conda activate paqui`
- Make sure you're in the `ui_local` directory 
- Install requirements: `pip install -r requirements.txt`

## Running the app
- Run `python app.py` to launch the app
- It'll print some address in command line, e.g., `http://127.0.0.1:5000`. Visit that address in browser to view

## Webpage order
- `prolific_id.html`
- `consent.html`
- `intro.html`
- `pre_survey.html` or `failure.html`
- `test_ui.html` or `model_rejection.html`
- `self_report.html`
- `comments.html`
- `thankyou.html`

