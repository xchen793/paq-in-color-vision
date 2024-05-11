// start colors = fixed colors in RGB 
const fixedColors = [
    'rgb(255, 0, 0)', // red (0.64, 0.33, 21.26)
    'rgb(0, 255, 0)', // green (0.3, 0.6, 71.52)
    'rgb(0, 0, 255)', // blue (0.15, 0.06, 7.22)
    'rgb(255, 255, 0)', // yellow (0.42, 0.51, 92.78)
    'rgb(255, 0, 255)'  // magenta (0.321, 0.154, 28.48)
];

// end colors
const endColors = [
    {x: 0.171, y: 0.0, Y: 1}, // tritan copunctal point
    {x: 0.747, y: 0.253, Y: 1}, // protan copunctal point
    {x: 1.080, y: -0.800, Y: 1} // deutan copunctal point
];

function parseRGB(rgbString) {
    const regex = /^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$/;
    const match = rgbString.match(regex);
    return match ? [parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10)] : null;
}

function RGBToxyY(rgbString) {
    const rgb = parseRGB(rgbString);
    if (!rgb) {
        console.error("Invalid RGB format");
        return null;
    }

    // Step 1: Convert RGB from 0-255 to 0-1
    let r = rgb[0] / 255;
    let g = rgb[1] / 255;
    let b = rgb[2] / 255;

    // Step 2: Apply reverse gamma correction (sRGB)
    r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Step 3: Apply the RGB to XYZ transformation matrix for sRGB D65
    const X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    const Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    const Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // Convert to xyY
    const sum = X + Y + Z;
    const x = sum === 0 ? 0 : X / sum;
    const y = sum === 0 ? 0 : Y / sum;
    return {x, y, Y};

}

const fixedColorsxyY = fixedColors.map(color => RGBToxyY(color));
console.log(fixedColorsxyY);

const colorPaths = [];

endColors.forEach(endxyY => {
    fixedColorsxyY.forEach(startxyY => {
        colorPaths.push({ startxyY, endxyY });
    });
});

console.log(colorPaths);

// Global variable to keep track of the current page
let currentPage = 1;  
let numPage = 15;
let totalsliderValue = 500;


document.addEventListener('DOMContentLoaded', function() {
    // Check if we are on the intro page
    if (document.getElementById('startButton')) {
        document.getElementById('startButton').addEventListener('click', function() {
            // Redirect to the first survey page
            window.location.href = 'survey_page1.html';
        });
    }

    // Check if the currentPage variable is defined
    if (typeof currentPage !== 'undefined') {
        currentPage = getPageNumberFromURL() || 1;
        history.replaceState({ page: currentPage }, `Page ${currentPage}`, `survey_page${currentPage}`);
        setupEventListeners();
        updateUIForPage(currentPage);
    }
});


function setupEventListeners() {
    const slider = document.getElementById("color-slider");
    const submitButton = document.getElementById("submit-color");
    const noMatchButton = document.getElementById("no-match");
    const nextPageButton = document.getElementById("next-page");

    // const slider = document.getElementById(`color-slider-${page}`);
    // const colorDisplay = document.getElementById(`color-display-${page}`);
    // const submitButton = document.getElementById(`submit-color-${page}`);
    // const noMatchButton = document.getElementById(`no-match-${page}`);
    // const nextPageButton = document.querySelector("[onclick^='window.location.href']");


    if (slider) {
        slider.addEventListener("input", function(){
            updateColor(slider.value);
        });
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
        nextPageButton.addEventListener("click", function(){
            goToNextPage();
        });
    } else {
        console.error("Next page button not found.");
    }
}

function getPageNumberFromURL() {
    const path = window.location.pathname.split('/');
    const pageSegment = path[path.length - 1];
    const match = pageSegment.match(/^survey_page(\d+)$/);
    return match ? parseInt(match[1], 10) : null;
}

function goToNextPage() {
    if (currentPage >= numPage){
        window.location.href = '/thankyou';
    } else{
        currentPage = currentPage + 1; // Loop from page 15 back to 1
        const nextPageUrl = `survey_page${currentPage}`;
        history.pushState({ page: currentPage }, `Page ${currentPage}`, nextPageUrl);
        updateUIForPage(currentPage);
    }

}

function updateUIForPage(page) {
    // Set the fixed color based on the current page
    const fixedColorDisplay = document.getElementById("fixed-color-display");
    if (fixedColorDisplay) {
        fixedColorDisplay.style.backgroundColor = fixedColors[((page - 1) % 5)];
    }
    // Reset the slider value to 0 for a fresh start on each page
    const slider = document.getElementById("color-slider");
    if (slider) {
        slider.value = 0;  // Reset slider to the start
        updateColor(0);  // Update the color based on the reset slider value
    } else {
        console.error("Slider not found.");
    }

    // Assuming there are elements to update per page, otherwise implement needed changes
    const colorDisplay = document.getElementById("color-display");
    updateColor(0);  // Initialize with a default position of the slider
}

function updateColor(sliderValue) {
    const colorDisplay = document.getElementById("color-display");
    let t = sliderValue / totalsliderValue;  // Normalize slider value to 0-1
    let { R, G, B } = interpolateColor(t, currentPage);
    colorDisplay.style.backgroundColor = `rgb(${R}, ${G}, ${B})`;
}

//Auxiliary Function 2: Get Page Id
function getPageId() {
    const path = window.location.pathname;
    pageId = path.substring(path.lastIndexOf('/') + 1) || 'survey_page0'
    return pageId;
}


function computeQueryVector(startxyY, endxyY) {
    if (!startxyY || !endxyY) {
        console.error('Invalid start or end xyY data.');
        return null;
    }
    const dx = endxyY.x - startxyY.x;
    const dy = endxyY.y - startxyY.y;
    const dY = endxyY.Y - startxyY.Y;
    const mag = Math.sqrt(dx * dx + dy * dy + dY * dY);
    return {
        x: dx / mag,
        y: dy / mag,
        Y: dY / mag
    };
}

function submitColor(actionType) {
    const slider = document.getElementById("color-slider");

    let currentPath = colorPaths[currentPage - 1];
    let query_vec = computeQueryVector(currentPath.startxyY, currentPath.endxyY);

    let fixedColorxyY = currentPath.startxyY;
    let endColorxyY = currentPath.endxyY;
    let pageId = getPageId();

    if (actionType === 'noMatch') {
        data = { 
            pageId: pageId, 
            query_vec: query_vec,
            gamma: null, 
            fixedColor: fixedColorxyY,
            endColor: endColorxyY
        };
    } else {
        data = {
            pageId: pageId,
            query_vec: query_vec,
            gamma: slider.value / totalsliderValue,
            fixedColor: fixedColorxyY,
            endColor: endColorxyY
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

/////////////////////////////////////  Converting functions xyY -> XYZ -> RGB /////////////////////////////////////
function xyYtoXYZ(x, y, Y) {
    if (y === 0) return { X: 0, Y: 0, Z: 0 };

    let X = (x * Y) / y;
    let Z = ((1 - x - y) * Y) / y;
    return { X, Y, Z };
}

function XYZtoRGB(X, Y, Z) {
    // Matrix transformation from XYZ to linear RGB
    let R =  3.2406*X - 1.5372*Y - 0.4986*Z;
    let G = -0.9689*X + 1.8758*Y + 0.0415*Z;
    let B =  0.0557*X - 0.2040*Y + 1.0570*Z;

    // Apply gamma correction and clamp the values between 0 and 1
    return {
        R: clampRGB(R),
        G: clampRGB(G),
        B: clampRGB(B)
    };
}

function clampRGB(value) {
    let linear = Math.max(0, Math.min(1, value));  // Clamp between 0 and 1
    if (linear <= 0.0031308)
        return 12.92 * linear;
    else
        return 1.055 * Math.pow(linear, 1/2.4) - 0.055;
}



function interpolateColor(t, page) {
    const path = colorPaths[page - 1];
    const startx = path.startxyY.x;
    const starty = path.startxyY.y;
    const startY = path.startxyY.Y;
    const endx = path.endxyY.x;
    const endy = path.endxyY.y;
    const endY = path.endxyY.Y;
    
    let x = startx + (endx - startx) * t;  // Linear interpolation from startX to endX
    let y = starty + (endy - starty) * t;  // Linear interpolation from startY to endY
    let Y = startY + (endY - startY) * t;  // Constant luminance

    // Convert to XYZ, then to RGB
    let { X, Y: newY, Z } = xyYtoXYZ(x, y, Y);
    let { R, G, B } = XYZtoRGB(X, newY, Z);

    // Convert 0-1 RGB to 0-255 for CSS usage
    return {
        R: R * 255,
        G: G * 255,
        B: B * 255
    };
}


