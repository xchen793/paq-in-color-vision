const luminance = 0.5;
const targetDistance = 0.05;
const numberOfDirections = 10;
const angleIncrement = 360 / numberOfDirections;
const max = 1; // slider threshold
const min = 0.7; // slider threshold
const numAddedpoints = 10;

let currentPage = 1;
let totalsliderValue = 100;
let threshold = 0.9;
let currentxyY = { x: 0, y: 0, Y: luminance };

// fixed colors in RGB for path 1
const fixedColors = [
    { x: 0.25, y: 0.34, Y: luminance },
    { x: 0.29, y: 0.40, Y: luminance },
    { x: 0.37, y: 0.40, Y: luminance },
    { x: 0.35, y: 0.35, Y: luminance }
];

// Convert degrees to radians
function degreesToRadians(degrees) {
    return degrees * (Math.PI / 180);
}


// endpoints 
const endpoints = [];
fixedColors.forEach(color => {
    // Retrieve the key for this color
    let colorKey = `x${color.x}-y${color.y}-Y${color.Y}`;
    for (let i = 0; i < numberOfDirections; i++) {
        const angle = i * angleIncrement;
        const { newX: newX1, newY: newY1 } = calculatePointOnDirection(color.x, color.y, angle, targetDistance);
        const { newX: newX2, newY: newY2 } = calculatePointOnDirection(color.x, color.y, angle, 0.5 * targetDistance);
        const { newX: newX3, newY: newY3 } = calculatePointOnDirection(color.x, color.y, angle, 0.33 * targetDistance);
        const vector = computeQueryVector({ x: newX1, y: newY1, Y: color.Y }, color);
        endpoints.push({ x: newX1, y: newY1, Y: luminance, flag: "fast", query_vec: vector});
        endpoints.push({ x: newX2, y: newY2, Y: luminance, flag: "medium", query_vec: vector });
        endpoints.push({ x: newX3, y: newY3, Y: luminance, flag: "slow", query_vec: vector});
    }
});


console.log("fixedcolor: ", fixedColors)

const duplicatedfixedColors = [];
// one quicker, one slower
fixedColors.forEach(color => {
    for (let i = 0; i < 3 * numberOfDirections; i++) {
        duplicatedfixedColors.push({ ...color }); // Use the spread operator to copy the object
    }
});

console.log("duplicatedfixedColors: ", duplicatedfixedColors)

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
    const flag = startxyY.flag;
    const query_vec = startxyY.query_vec
    colorPaths.push({ startxyY, endxyY, flag, query_vec });
}

//////////////////////// Build Color Path2 ////////////////////////
const steps = numAddedpoints;

// Convert each fixed color from xyY to RGB
const rgbColors = fixedColors.map(color => {
    const { X, Y, Z } = xyYtoXYZ(color.x, color.y, color.Y);
    return {
        R: parseRGB(XYZtoRGB(X, Y, Z))[0],
        G: parseRGB(XYZtoRGB(X, Y, Z))[1],
        B: parseRGB(XYZtoRGB(X, Y, Z))[2]
    };
});

function createRandomTargetColor(color) {
    const variation = 40;  // Maximum variation added to each color component
    return {
        R: clamp(color.R + variation, 0, 255),
        G: clamp(color.G + variation, 0, 255),
        B: clamp(color.B + variation, 0, 255)
    };
}

// Interpolate color path in RGB 
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

// Shuffle arrays concurrently
function shuffleArrays(array1, array2, array3) {
    // Combine the arrays into an array of triples
    let combined = array1.map((value, index) => [value, array2[index], array3[index]]);

    // Shuffle the combined array
    for (let i = combined.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [combined[i], combined[j]] = [combined[j], combined[i]];
    }

    // Separate the combined array back into three arrays
    array1 = combined.map(triple => triple[0]);
    array2 = combined.map(triple => triple[1]);
    array3 = combined.map(triple => triple[2]);

    return [array1, array2, array3];
}

// Shuffle the duplicatedfixedColors, colorPaths, and colorPaths2 concurrently
let [shuffledFixedColors, shuffledColorPaths, shuffledColorPaths2] = shuffleArrays(duplicatedfixedColors, colorPaths, colorPaths2);
const shuffledFixedColors_rgb = shuffledFixedColors.map(color => {
    const { X, Y, Z } = xyYtoXYZ(color.x, color.y, color.Y);
    return XYZtoRGB(X, Y, Z);
});


document.addEventListener('DOMContentLoaded', function () {
    if (typeof currentPage !== 'undefined') {
        currentPage = getPageNumberFromURL() || 1;
        history.replaceState({ page: currentPage }, `Page ${currentPage}`, `survey_page${currentPage}`);
        setupEventListeners();
        updateUIForPage(currentPage);
    }
});

function setupEventListeners() {
    const slider = document.getElementById("color-slider");
    const nextPageButton = document.getElementById("next-page");

    if (slider) {
        slider.addEventListener("input", function () {
            updateColor(slider.value);
        });

        slider.addEventListener("change", function () {
            updateColor(slider.value);
        });
    } else {
        console.error("Slider not found.");
    }

    if (nextPageButton) {
        nextPageButton.addEventListener("click", function () {
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
    if (currentPage >= numPage) {
        window.location.href = '/thankyou';
    } else {
        currentPage = currentPage + 1;
        const nextPageUrl = `survey_page${currentPage}`;
        history.pushState({ page: currentPage }, `Page ${currentPage}`, nextPageUrl);
        updateUIForPage(currentPage);
    }
}

function updateUIForPage(page) {
    const fixedColorDisplay = document.getElementById("fixed-color-display");
    if (fixedColorDisplay) {
        fixedColorDisplay.style.backgroundColor = shuffledFixedColors_rgb[(page - 1)];
    }
    const slider = document.getElementById("color-slider");
    if (slider) {
        slider.value = 0;
        updateColor(0);
    } else {
        console.error("Slider not found.");
    }
    threshold = Math.random() * (max - min) + min;
    updateColor(0);
}

function updateColor(sliderValue) {
    const colorDisplay = document.getElementById("color-display");

    let t = sliderValue / totalsliderValue;

    if (t <= threshold) {
        t = t / threshold;
        let color = interpolateColor(t, currentPage);
        colorDisplay.style.backgroundColor = `rgb(${color.R}, ${color.G}, ${color.B})`;
    } else {
        t = (t - threshold) / (1 - threshold);
        let index = Math.floor(t * numAddedpoints);
        index = Math.min(index, numAddedpoints - 1);

        if (shuffledColorPaths2[currentPage - 1] && shuffledColorPaths2[currentPage - 1][index]) {
            let color = shuffledColorPaths2[currentPage - 1][index];
            colorDisplay.style.backgroundColor = color;
        } else {
            console.error("Color path data not found for index:", index);
        }
    }
}

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

    let r = rgb[0] / 255;
    let g = rgb[1] / 255;
    let b = rgb[2] / 255;

    r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    const X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    const Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    const Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    const sum = X + Y + Z;
    const x = sum === 0 ? 0 : X / sum;
    const y = sum === 0 ? 0 : Y / sum;
    return { x, y, Y };
}

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
        return null;
    }
    const dx = endxyY.x - startxyY.x;
    const dy = endxyY.y - startxyY.y;
    const dY = endxyY.Y - startxyY.Y;
    const distance = Math.sqrt(dx * dx + dy * dy + dY * dY);
    return distance;
}

// Calculate point on direction vector at a certain distance
function calculatePointOnDirection(x, y, angle, distance) {
    const angleRadians = degreesToRadians(angle);
    const newX = x + distance * Math.cos(angleRadians);
    const newY = y + distance * Math.sin(angleRadians);
    return { newX, newY };
}


function submitColor(actionType) {
    const slider = document.getElementById("color-slider");

    let currentPath = shuffledColorPaths[currentPage - 1];
    let gamma = computeDistance(currentPath.endxyY, currentxyY);

    let fixedColorxyY = currentPath.endxyY;
    let startColorxyY = currentPath.startxyY;
    let flag = currentPath.flag;  // Include flag in the data
    let query_vec = currentPath.query_vec
    let pageId = getPageId();

    let data;

    if (actionType === 'noMatch') {
        data = {
            pageId: pageId,
            query_vec: query_vec,
            gamma: null,
            fixedColor: fixedColorxyY,
            startColor: startColorxyY,
            flag: flag  // Include flag in the data
        };
    } else {
        data = {
            pageId: pageId,
            query_vec: query_vec,
            gamma: gamma,
            fixedColor: fixedColorxyY,
            endColor: startColorxyY,
            flag: flag  // Include flag in the data
        };
    }

    fetch(`/submit-color`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(data => {
            console.log(`Response submitted: ${data.status}`); // Log the response instead of showing an alert
        })
        .catch(error => console.error('Error submitting response:', error));
}

function xyYtoXYZ(x, y, Y) {
    if (y === 0) return { X: 0, Y: 0, Z: 0 };

    const X = (x * Y) / y;
    const Z = ((1 - x - y) * Y) / y;
    return { X, Y, Z };
}

function XYZtoRGB(X, Y, Z) {
    let R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    let G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    let B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;

    R = gammaCorrection(R);
    G = gammaCorrection(G);
    B = gammaCorrection(B);

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

function interpolateColor(t, page) {
    const path = shuffledColorPaths[page - 1];

    if (!path || !path.endxyY) {
        console.error("Invalid path or endxyY data at page: ", page, " path: ", path);
        return { R: 0, G: 0, B: 0 };
    }

    const startx = path.startxyY.x;
    const starty = path.startxyY.y;
    const startY = path.startxyY.Y;
    const endx = path.endxyY.x;
    const endy = path.endxyY.y;
    const endY = path.endxyY.Y;

    currentxyY.x = startx + (endx - startx) * t;
    currentxyY.y = starty + (endy - starty) * t;
    currentxyY.Y = startY + (endY - startY) * t;

    let { X, Y: newY, Z } = xyYtoXYZ(currentxyY.x, currentxyY.y, currentxyY.Y);

    return {
        R: parseRGB(XYZtoRGB(X, newY, Z))[0],
        G: parseRGB(XYZtoRGB(X, newY, Z))[1],
        B: parseRGB(XYZtoRGB(X, newY, Z))[2]
    };
}



