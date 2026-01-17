import {RGB2DImage, numberVector, number3DMatrix, undefinedDimensionNumberArray} from "./types/types";

function getUpperBoundSqrt(n: number): number {
    return Math.ceil(Math.sqrt(n));
}

export function getEmptyRGB2DImage(sizeX: number, sizeY: number): RGB2DImage {
    return Array.from({length: sizeX}, () =>
        Array.from({length: sizeY}, () => null)
    );
}

export function getDefaultLatentVector(size: number): numberVector {
    return Array(size).fill(0);
}

function getRGB2DImageFromVector(inputVector: numberVector, min: number, max: number): RGB2DImage {
    const newDimension = getUpperBoundSqrt(inputVector.length);

    const result = getEmptyRGB2DImage(newDimension, newDimension);

    let tempValue: number;
    for (let i = 0; i < newDimension; i++) {
        for (let j = 0; j < newDimension; j++) {
            if (i * newDimension + j < inputVector.length) {
                tempValue = mapTo255(min, max, inputVector[i * newDimension + j]);
                result[i][j] = [tempValue, tempValue, tempValue];
            }
        }
    }

    return result;
}

function getResultFrom3DCase(content: number3DMatrix, minimum: number, maximum: number): RGB2DImage {
    //todo maybe use three js here ?
    //or slider to have several 2d pictures and allowed to go trough the frames
    const outerDimension = getUpperBoundSqrt(content[0][0].length);
    const innerDimension = content.length;
    const margin = Math.ceil(10 / outerDimension); //margin depends on the number of "smaller images"

    const result = getEmptyRGB2DImage(outerDimension * (innerDimension + margin), outerDimension * (innerDimension + margin));

    let tempValue: number;

    for (let outerX = 0; outerX < outerDimension; outerX++) {
        for (let outerY = 0; outerY < outerDimension; outerY++) {

            for (let innerX = 0; innerX < innerDimension; innerX++) {
                for (let innerY = 0; innerY < innerDimension; innerY++) {

                    if (outerX * outerDimension + outerY < content[0][0].length) {
                        // 255 white --- 0 black
                        tempValue = mapTo255(minimum, maximum, content[innerX][innerY][outerX * outerDimension + outerY]);
                        result[outerX * (innerDimension + margin) + innerX][outerY * (innerDimension + margin) + innerY] = [tempValue, tempValue, tempValue];
                    }
                }
            }
        }
    }

    return result;
}

export function getRGB2DImageFromRawContent(rawContent: undefinedDimensionNumberArray): RGB2DImage {

    const dimensionsQuantity = getArrayDimensions(rawContent);

    const minimum = getOverallMinimum(rawContent);
    const maximum = getOverallMaximum(rawContent);

    let result: RGB2DImage;

    if (dimensionsQuantity === 1) {
        result = getRGB2DImageFromVector(rawContent as numberVector, minimum, maximum);

    } else if (dimensionsQuantity === 3) {
        result = getResultFrom3DCase(rawContent as number3DMatrix, minimum, maximum);

    } else {
        console.log('ERROR');
    }

    return result;
}

function getArrayDimensions(array: undefinedDimensionNumberArray): number {
    let dimensionsQuantity = 0;
    let cursor: any = array;

    while (Array.isArray(cursor)) {
        dimensionsQuantity++;
        cursor = cursor[0];
    }

    return dimensionsQuantity;
}

function getOverallMinimum(array: undefinedDimensionNumberArray): number {//todo maybe split for the two dimensions possible ?
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

function getOverallMaximum(array: undefinedDimensionNumberArray): number {
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

export function convertNumberToStringPercentage(value: number): string {
    return (value * 100).toFixed(2) + "%";
}

function projectTo255(x: number, maxVisualizationInput: number): number { //todo remove the other 255 func
    const clampedVal = Math.min(Math.max(x, -maxVisualizationInput), maxVisualizationInput);
    return ((clampedVal + maxVisualizationInput) / (2 * maxVisualizationInput)) * 255;
}

function mapTo255(minValue: number, maxValue: number, value: number): number {
    const clampedVal = Math.min(Math.max(value, minValue), maxValue);
    const ratio = (clampedVal - minValue) / (maxValue - minValue);
    return Math.round(ratio * 255);
}

export function getInputVectorAsRGB2DImage(inputVector: numberVector, size: number, maxVisualizationInput: number): RGB2DImage {
    const latentVectorAsMatrix: RGB2DImage = getEmptyRGB2DImage(size, size);

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const intensityProjected: number = projectTo255(inputVector[i * size + j], maxVisualizationInput); //between 0 black and 255 white

            latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
        }
    }
    return latentVectorAsMatrix;
}

export function getRandomNormalFloat(mu: number, sigma: number): number {
    const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
    return mu + sigma * z;
}