
const luminance = 0.5;
const targetDistance = 0.05;
const numberOfDirections = 20;
const angleIncrement = 360 / numberOfDirections;
const max = 1; //slider threshold
const min = 0.7; //slider threshold
const numAddedpoints = 10;

let currentPage = 1;  

let totalsliderValue = 100;


let threshold = 0.9;
let currentxyY = {x: 0, y: 0, Y: luminance};  

// fixed colors in RGB for path 1
const fixedColors = [
    {x: 0.25, y: 0.34, Y: luminance}, 
    {x:0.29, y: 0.40, Y: luminance}, 
    {x: 0.37, y: 0.40, Y: luminance},
    {x: 0.35, y: 0.35, Y: luminance} 
];


// Convert degrees to radians
function degreesToRadians(degrees) {
    return degrees * (Math.PI / 180);
}

// Calculate point on direction vector at a certain distance
function calculatePointOnDirection(x, y, angle, distance) {
    const angleRadians = degreesToRadians(angle);
    const newX = x + distance * Math.cos(angleRadians);
    const newY = y + distance * Math.sin(angleRadians);
    return { newX, newY };

}

//endpoints 
const endpoints = [];

fixedColors.forEach((color, index) => {
    for (let i = 0; i < numberOfDirections; i++) {
        const angle = i * angleIncrement;
        const { newX, newY } = calculatePointOnDirection(color.x, color.y, angle, targetDistance);
        endpoints.push({ x: newX, y: newY, Y: luminance});
    }
});

const duplicatedfixedColors = [];

fixedColors.forEach(color => {
    for (let i = 0; i < numberOfDirections; i++) {
        duplicatedfixedColors.push({ ...color }); // Use the spread operator to copy the object
    }
});

let numPage = duplicatedfixedColors.length; // Increased granularity for the slider

const duplicatedfixedColors_rgb = duplicatedfixedColors.map(color => {
    const { X, Y, Z } = xyYtoXYZ(color.x, color.y, color.Y);
    return XYZtoRGB(X, Y, Z);
});

//////////////////////// Build the color paths /////////////////////////
const colorPaths = [];

for (let i = 0; i < endpoints.length; i++) {
    const endxyY = duplicatedfixedColors[i];
    const startxyY = endpoints[i];
    
    colorPaths.push({ startxyY, endxyY });
    
}

console.log("colorPaths: ", colorPaths);


////////////////////////// Global variable to keep track of the current page //////////////////////////


console.log("num page: ", numPage)
////////////////////////// Functions //////////////////////////
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



document.addEventListener('DOMContentLoaded', function() {
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
    const noMatchButton = document.getElementById("no-match");
    const nextPageButton = document.getElementById("next-page");

    if (slider) {
        // Update color in real-time as the slider is dragged
        slider.addEventListener("input", function() {
            updateColor(slider.value);
        });

        // Optional: Update color when the user releases the slider, ensuring final color is set
        slider.addEventListener("change", function() {
            updateColor(slider.value);
        });
    } else {
        console.error("Slider not found.");
    } 

    if (noMatchButton) {
        noMatchButton.addEventListener("click", function() {
            submitColor('noMatch');
            goToNextPage(); // Navigate to the next page after submitting
        });
    } else {
        console.error("No Match button not found.");
    }

    if (nextPageButton) {
        nextPageButton.addEventListener("click", function() {
            submitColor('submitColor');
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
        currentPage = currentPage + 1; 
        const nextPageUrl = `survey_page${currentPage}`;
        history.pushState({ page: currentPage }, `Page ${currentPage}`, nextPageUrl);
        updateUIForPage(currentPage);
    }

}

function updateUIForPage(page) {
    // Set the fixed color based on the current page
    const fixedColorDisplay = document.getElementById("fixed-color-display");
    if (fixedColorDisplay) {
        fixedColorDisplay.style.backgroundColor = duplicatedfixedColors_rgb[(page - 1)];
    }
    // Reset the slider value to 0 for a fresh start on each page
    const slider = document.getElementById("color-slider");
    if (slider) {
        slider.value = 0;  // Reset slider to the start
        updateColor(0);  // Update the color based on the reset slider value
    } else {
        console.error("Slider not found.");
    }
    threshold = Math.random() * (max - min) + min;
    console.log("threshold: ", threshold);
    updateColor(0);  // Initialize with a default position of the slider
}

function updateColor(sliderValue) {
    const colorDisplay = document.getElementById("color-display");
    
    // Normalize slider value to 0-1
    let t = sliderValue / totalsliderValue;
    
    if (t <= threshold) {
        t = t / threshold; 
        console.log("First color path active, normalized t:", t);
        console.log("currentxyY", currentxyY);
        ({ R, G, B } = interpolateColor(t, currentPage));
        colorDisplay.style.backgroundColor = `rgb(${R}, ${G}, ${B})`;
    } else {
        // Normalize for the second segment
        t = (t - threshold) / (1 - threshold);
        console.log("Second color path active, normalized t:", t);
        console.log("currentxyY", currentxyY);
        // Calculate index within the range of available colors in colorPaths2
        let index = Math.floor(t * numAddedpoints);  // Calculate index in the 10 color points
        index = Math.min(index, numAddedpoints - 1); // Ensure index does not exceed bounds
        
        // Fetch color from colorPaths2 for the current page
        console.log("colorPaths2[currentPage - 1]: ", colorPaths2[currentPage - 1]);
        console.log("colorPaths2[currentPage - 1][index]: ", colorPaths2[currentPage - 1][index]);
        if (colorPaths2[currentPage - 1] && colorPaths2[currentPage - 1][index]) {
            let color = colorPaths2[currentPage - 1][index]; // This should directly be a 'rgb(R, G, B)' string
            colorDisplay.style.backgroundColor = color; // Directly use the RGB string
        } else {
            console.error("Color path data not found for index:", index);
        }
    }
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

function computeDistance(startxyY, endxyY) {
    if (!startxyY || !endxyY) {
        console.error('Invalid start or end xyY data.');
        return null;  // Return null to indicate an error.
    }
    // Calculate the differences in each coordinate
    const dx = endxyY.x - startxyY.x;
    const dy = endxyY.y - startxyY.y;
    const dY = endxyY.Y - startxyY.Y;
    // Compute the Euclidean distance using the Pythagorean theorem
    const distance = Math.sqrt(dx * dx + dy * dy + dY * dY);
    return distance;
}


function submitColor(actionType) {
    const slider = document.getElementById("color-slider");

    let currentPath = colorPaths[currentPage - 1];
    let query_vec = computeQueryVector(currentPath.endxyY, currentPath.startxyY); //refpt -> startpoint
    let gamma = computeDistance(currentPath.endxyY, currentxyY); //gamma = eucdis(refpt, curpt)

    let fixedColorxyY = currentPath.endxyY;
    let startColorxyY = currentPath.startxyY;
    let pageId = getPageId();

    if (actionType === 'noMatch') {
        data = { 
            pageId: pageId, 
            query_vec: query_vec,
            gamma: null, 
            fixedColor: fixedColorxyY,
            startColor: startColorxyY
        };
    } else {
        data = {
            pageId: pageId,
            query_vec: query_vec,
            gamma: gamma,
            fixedColor: fixedColorxyY,
            endColor: startColorxyY
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

    const X = (x * Y) / y;
    const Z = ((1 - x - y) * Y) / y;
    return { X, Y, Z };
}

function XYZtoRGB(X, Y, Z) {
    // Matrix transformation from XYZ to linear RGB (sRGB)
    let R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    let G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    let B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;

    // Apply gamma correction
    R = gammaCorrection(R);
    G = gammaCorrection(G);
    B = gammaCorrection(B);

    // Clamp the values between 0 and 255 for RGB
    R = Math.round(clamp(R * 255, 0, 255));
    G = Math.round(clamp(G * 255, 0, 255));
    B = Math.round(clamp(B * 255, 0, 255));

    return `rgb(${R}, ${G}, ${B})`;
}

function gammaCorrection(value) {
    return value <= 0.0031308 ? 12.92 * value : 1.055 * Math.pow(value, 1 / 2.4) - 0.055;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}



//////////////////// Slider Color Interpolation ///////////////////////
function interpolateColor(t, page) {
    const path = colorPaths[page - 1];

    const startx = path.startxyY.x;
    const starty = path.startxyY.y;
    const startY = path.startxyY.Y;
    const endx = path.endxyY.x;
    const endy = path.endxyY.y;
    const endY = path.endxyY.Y;
    // console.log(path)
    currentxyY.x = startx + (endx - startx) * t;  
    currentxyY.y = starty + (endy - starty) * t;  
    currentxyY.Y = startY + (endY - startY) * t;  


    // Convert to XYZ, then to RGB
    let {X, Y:newY, Z} = xyYtoXYZ(currentxyY.x, currentxyY.y, currentxyY.Y);


    // Convert 0-1 RGB to 0-255 for CSS usage
    return {
        R: parseRGB(XYZtoRGB(X, newY, Z))[0],
        G: parseRGB(XYZtoRGB(X, newY, Z))[1],
        B: parseRGB(XYZtoRGB(X, newY, Z))[2]
    };
}


//////////////////////// Build Color Path2 ////////////////////////
const steps = numAddedpoints;

// Convert each fixed color from xyY to RGB
const rgbColors = fixedColors.map(color => {
    const {X, Y, Z} = xyYtoXYZ(color.x, color.y, color.Y);
    return {
        R: parseRGB(XYZtoRGB(X, Y, Z))[0],
        G: parseRGB(XYZtoRGB(X, Y, Z))[1],
        B: parseRGB(XYZtoRGB(X, Y, Z))[2]
    };
});
console.log("rgbColors: ", rgbColors);

function createRandomTargetColor(color) {
    const variation = 40;  // Maximum variation added to each color component
    return {
        R: clamp(color.R + variation, 0, 255),
        G: clamp(color.G + variation, 0, 255),
        B: clamp(color.B + variation, 0, 255)
    };
}

//interpolate color path in RGB 
function interpolateColor2(startColor, endColor, steps) {
    let colorPath = [];
    for (let i = 0; i < steps; i++) {
        let t = i / (steps - 1);
        let interpolatedColor = {
            R: Math.round(startColor.R + (endColor.R - startColor.R) * t),
            G: Math.round(startColor.G + (endColor.G - startColor.G) * t),
            B: Math.round(startColor.B + (endColor.B - startColor.B) * t)
        };
        colorPath.push(`rgb(${interpolatedColor.R}, ${interpolatedColor.G}, ${interpolatedColor.B})`);
    }
    return colorPath;
}

// Generate target colors for each RGB color with randomness
const targetColors = [
    { R: 0, G: 128, B: 128 },
    { R: 0, G: 139, B: 139 },
    { R: 184, G: 134, B: 11 },
    { R: 255, G: 105, B: 180 }
];

// Generate color paths for each pair of start and target colors
const colorPaths22 = rgbColors.map((rgbColor, index) => interpolateColor2(rgbColor, targetColors[index], steps));

// Duplicate each color path row 20 times to construct colorPaths2
const colorPaths2 = colorPaths22.flatMap(path => Array(20).fill([...path]));







