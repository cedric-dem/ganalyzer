import {ImageGridRenderer} from "./renderers.js";

export function changeInsideRepresentation(content, location) {
    //empty previous content if any
    document.getElementById(location).innerHTML = "";

    // get matrix
    const matrixReadyToDisplay = getMatrixToDisplay(content);

    // create pixels
    const imageGridRenderer = new ImageGridRenderer()
    const locationImage = imageGridRenderer.initializeImage(location, matrixReadyToDisplay.length, matrixReadyToDisplay[0].length);

    // colour pixels
    imageGridRenderer.changeImage(matrixReadyToDisplay, locationImage);
}

function getUpperBoundSqrt(n) {
    return Math.ceil(Math.sqrt(n));
}

export function getNull2DArray(sizeX, sizeY) {
    return Array.from({length: sizeX}, () =>
        Array.from({length: sizeY}, () => null)
    );
}

function getResultFrom1DCase(content, min, max) {
    const newDimension = getUpperBoundSqrt(content.length)

    const result = getNull2DArray(newDimension, newDimension)

    let tempValue;
    for (let i = 0; i < newDimension; i++) {
        for (let j = 0; j < newDimension; j++) {
            if (i * newDimension + j < content.length) {
                tempValue = mapTo255(min, max, content[i * newDimension + j]);
                result[i][j] = [tempValue, tempValue, tempValue];
            }
        }
    }

    return result;
}

function getResultFrom3DCase(content, minimum, maximum) {
    //todo maybe use three js here ?
    //or slider to have several 2d pictures and allowed to go trough the frames
    const outerDimension = getUpperBoundSqrt(content[0][0].length)
    const innerDimension = content.length
    const margin = Math.ceil(10 / outerDimension); //margin depends on the number of "smaller images"

    const result = getNull2DArray(outerDimension * (innerDimension + margin), outerDimension * (innerDimension + margin));

    let tempValue;

    for (let outerX = 0; outerX < outerDimension; outerX++) {
        for (let outerY = 0; outerY < outerDimension; outerY++) {

            for (let innerX = 0; innerX < innerDimension; innerX++) {
                for (let innerY = 0; innerY < innerDimension; innerY++) {

                    if (outerX * outerDimension + outerY < content[0][0].length) {
                        // 255 white --- 0 black
                        tempValue = mapTo255(minimum, maximum, content[innerX][innerY][outerX * outerDimension + outerY]);
                        result[outerX * (innerDimension + margin) + innerX][outerY * (innerDimension + margin) + innerY] = [tempValue, tempValue, tempValue]
                    }
                }
            }
        }
    }

    return result;
}

export function getMatrixToDisplay(rawContent) {

    const dimensionsQuantity = getArrayDimensions(rawContent);

    const minimum = getOverallMinimum(rawContent);
    const maximum = getOverallMaximum(rawContent);

    let result;

    if (dimensionsQuantity === 1) {
        result = getResultFrom1DCase(rawContent, minimum, maximum);

    } else if (dimensionsQuantity === 3) {
        result = getResultFrom3DCase(rawContent, minimum, maximum);

    } else {
        console.log('ERROR')
    }

    return result
}

function getArrayDimensions(array) {
    let dimensionsQuantity = 0;

    while (Array.isArray(array)) {
        dimensionsQuantity++;
        array = array[0];
    }

    return dimensionsQuantity;
}

function getOverallMinimum(array) {
    let minimum = Infinity;
    for (const element of array) {
        if (Array.isArray(element)) {
            minimum = Math.min(minimum, getOverallMinimum(element));
        } else {
            minimum = Math.min(minimum, element);
        }
    }
    return minimum;
}

function getOverallMaximum(array) {
    let maximum = -Infinity;
    for (const element of array) {
        if (Array.isArray(element)) {
            maximum = Math.max(maximum, getOverallMaximum(element));
        } else {
            maximum = Math.max(maximum, element);
        }
    }
    return maximum;
}

export function addChoices(controller, isGenerator, location, layersList) { //todo coulr remove boolean isgenerator
    const select = document.getElementById(location);

    layersList.forEach((layer) => {
        const option = new Option(layer);
        select.add(option);
    });

    select.addEventListener("change", () => {
        const newLayerName = select.value;

        controller.refreshInside(newLayerName)
    });
}

export function toPercentage(value) {
    return (value * 100).toFixed(2) + "%";
}

function projectTo255(x, maxVisualizationInput) { //todo remove the other 255 func
    const clampedVal = Math.min(Math.max(x, -maxVisualizationInput), maxVisualizationInput);
    return ((clampedVal + maxVisualizationInput) / (2 * maxVisualizationInput)) * 255;
}

function mapTo255(minValue, maxValue, value) {
    const clampedVal = Math.min(Math.max(value, minValue), maxValue);
    const ratio = (clampedVal - minValue) / (maxValue - minValue);
    return Math.round(ratio * 255);
}

export function getInputVectorAsMatrix(inputVector, size, maxVisualizationInput) {
    const latentVectorAsMatrix = getNull2DArray(size, size)

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const intensityProjected = projectTo255(inputVector[i * size + j], maxVisualizationInput); //between 0 black and 255 white

            latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
        }
    }
    return latentVectorAsMatrix;
}

export function getRandomNormalFloat(mu, sigma){
    const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
    return mu + sigma * z;
}
