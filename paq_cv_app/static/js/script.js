// Define a customizable path through RGB space as an array of [R, G, B] points
const colorPath = [
    [0, 0, 0],     // Black
    [255, 0, 0],   // Red
    [255, 255, 0], // Yellow
    [0, 255, 0],   // Green
    [0, 255, 255], // Cyan
    [0, 0, 255],   // Blue
    [255, 0, 255], // Magenta
    [255, 255, 255]// White
];


document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners(); 
});



function setupEventListeners() {
    const slider = document.getElementById("color-slider");
    const submitButton = document.getElementById("submit-color");
    const noMatchButton = document.getElementById("no-match");
    const nextPageButton = document.getElementById("next-page");

    if (slider) {
        slider.addEventListener("input", updateColor);
    } else {
        console.error("Slider not found.");
    }

    if (submitButton) {
        submitButton.addEventListener("click", function() {
            submitColor('submitColor');
        });
    } else {
        console.error("Submit button not found.");
    }

    if (noMatchButton) {
        noMatchButton.addEventListener("click", function() {
            submitColor('noMatch');
        });
    } else {
        console.error("No Match button not found.");
    }

    if (nextPageButton) {
        nextPageButton.addEventListener("click", goToNextPage);
    } else {
        console.error("Next page button not found.");
    }
}

// Auxiliary Function 1: Parse RGB string 
function parseRGB(rgbString) {
    // Regular expression to extract numbers from the rgb string
    const regex = /^rgb\(\s*(\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\s*\)$/;
    const match = rgbString.match(regex);
    
    const red = parseInt(match[1], 10)
    const green = parseInt(match[2], 10)
    const blue = parseInt(match[3], 10)

    if (match) {
        return {
            r: red, // Convert the first captured group to an integer (Red)
            g: green, // Convert the second captured group to an integer (Green)
            b: blue  // Convert the third captured group to an integer (Blue)
        };
    } else {
        console.error("Invalid RGB format:", rgbString);
        return null; // Return null if the format doesn't match
    }
}

//Auxiliary Function 2: Get Page Id
function getPageId() {
    const path = window.location.pathname;
    pageId = path.substring(path.lastIndexOf('/') + 1) || 'survey_page0'
    return pageId;
}

function submitColor(actionType) {
    const colorDisplay = document.getElementById("color-display");
    const fixedColorDisplay = document.getElementById("fixed-color-display");
    const xyData = colorDisplay.getAttribute('data-xy'); // Ensure this is set correctly

    const rgbString = window.getComputedStyle(fixedColorDisplay).backgroundColor;
    const rgb = parseRGB(rgbString);

    if (!rgb) {
        console.error("Failed to parse RGB. Submission aborted.");
        return;
    }

    const fixedColorXyY = rgbToXyY(rgb.r, rgb.g, rgb.b);
    const pageId = getPageId();

    let data; // Use let to allow modification
    if (actionType === 'noMatch') {
        data = { 
            pageId: pageId, 
            xyData: null, 
            fixedColor: fixedColorXyY 
        };
    } else {
        data = {
            pageId: pageId,
            xyData: JSON.parse(xyData || '{}'), // Safely parse xyData or default to {}
            fixedColor: fixedColorXyY
        };
    }

    fetch(`/submit-color`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => alert(`Response submitted: ${data.status}`))
    .catch(error => console.error('Error submitting response:', error));
}

// This is the mapping !!!
function interpolateColor(path, sliderPosition) {
    // Determine the interval in the path
    const numIntervals = path.length - 1;
    const intervalSize = 100 / numIntervals;
    const index = Math.min(Math.floor(sliderPosition / intervalSize), numIntervals - 1);

    // Compute the local position within the interval
    const localPosition = (sliderPosition % intervalSize) / intervalSize;

    // Interpolate between the current and next point in the path
    const startPoint = path[index];
    const endPoint = path[index + 1];
    const interpolatedColor = startPoint.map((start, i) => {
        const end = endPoint[i];
        return Math.round(start + (end - start) * localPosition);
    });

    return interpolatedColor;
}

function rgbToXyY(r, g, b) {
    // Normalize RGB values to the range 0-1
    r = r / 255;
    g = g / 255;
    b = b / 255;
    // First, inverse the gamma correction (sRGB)
    r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Apply wide gamut RGB D65 conversion coefficients
    const X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    const Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    const Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // Convert to xyY
    const sum = X + Y + Z;
    const x = sum === 0 ? 0 : X / sum;
    const y = sum === 0 ? 0 : Y / sum;
    return {x, y, Y};
}


function updateColor() {
    const slider = document.getElementById("color-slider");
    const colorDisplay = document.getElementById("color-display");
    if (slider && colorDisplay) {
        const sliderValue = slider.value;
        const newColor = interpolateColor(colorPath, sliderValue);
        colorDisplay.style.backgroundColor = `rgb(${newColor[0]}, ${newColor[1]}, ${newColor[2]})`;
        const xyY = rgbToXyY(newColor[0] / 255, newColor[1] / 255, newColor[2] / 255);
        colorDisplay.setAttribute('data-xy', JSON.stringify(xyY));
    }
}


function goToNextPage() {
    console.log("Current Page:", currentPage); // This will show you what is being captured as currentPage
    var pathname = window.location.pathname;
    var currentPage = pathname.split('/').pop();

    if (currentPage === '') {
        currentPage = 'test_ui';  // Adjust if your root goes to a different page
    }
    switch(currentPage) {
        case 'test_ui':
            window.location.href = '/survey_page1';
            break;
        case 'survey_page1':
            window.location.href = '/survey_page2';
            break;
        case 'survey_page2':
            window.location.href = '/survey_page3';
            break;
        case 'survey_page3':
            window.location.href = '/survey_page4';
            break;
        case 'survey_page4':
            window.location.href = '/survey_page5';
            break;
        case 'survey_page5':
            window.location.href = '/thankyou';
            break;
        default:
            alert('You are at the end of the survey.');
            break;
    }
}

// Initialize with default color
updateColor();



