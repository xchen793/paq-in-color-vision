4/29: Local webapp that picks a random starting color and direction in RGB space and lets users drag along that direction.

# Getting started
This was tested with python 3.11.5. 
- Create a new conda environment: `conda create --name paqui -c conda-forge python=3.11.5`
- Activate environment: `conda activate paqui`
- Make sure you're in the `ui_local` directory 
- Install requirements: `pip install -r requirements.txt`

# Running the app
- Run `python app.py` to launch the app
- It'll print some address in command line, e.g., `http://127.0.0.1:5000`. Visit that address in browser to view

# Future to-dos
- [x] Add reference color
- [ ] Get scaling right: Currently, the slider lets users slide from 0 - 500. However, depending on the magnitude of the randomly chosen direction, this may not encompass all colors along that direction.
- [ ] Store PAQ response: Need to return slider value back to webapp for future processing 
- [ ] Build a full user experience: Need to run this loop (pick ref and direction, render page, store user response) sequentially