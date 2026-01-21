import {RGB2DImage, numberVector, number3DMatrix} from "./types/types";

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
    //console.log("1D case")
    const newDimension = getUpperBoundSqrt(inputVector.length);

    const result = getEmptyRGB2DImage(newDimension, newDimension);

    let tempValue: number;
    for (let i = 0; i < newDimension; i++) {
        for (let j = 0; j < newDimension; j++) {
            if (i * newDimension + j < inputVector.length) {
                if (min === max) {
                    tempValue = 127;
                } else {
                    tempValue = mapTo255(min, max, inputVector[i * newDimension + j]);
                }
                result[i][j] = [tempValue, tempValue, tempValue];
            }
        }
    }

    return result;
}

function getResultFrom3DCase(content: number3DMatrix, minimum: number, maximum: number): RGB2DImage {
    //console.log("3D Case")
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

export function getRGB2DImageFromRawContent(rawContent: number3DMatrix): RGB2DImage {
    let minimum = null
    let maximum = null;

    [minimum, maximum] = getOverallMinimumAndMaximum(rawContent);

    let result: RGB2DImage;

    if (rawContent.length == 1 && rawContent[0].length == 1 && rawContent[0][0].length == 1) {
        //only one digit
        //result = getRGB2DImageFromVector(rawContent[0][0], minimum, maximum);

    } else if (rawContent.length == 1 && rawContent[0].length == 1) {
        // 2d array
        result = getRGB2DImageFromVector(rawContent[0][0], minimum, maximum);

    } else if (rawContent.length == 1) {
        //nothing ?

    } else {
        result = getResultFrom3DCase(rawContent, minimum, maximum);
    }
    return result;
}

export function getOverallMinimumAndMaximum(arr: number3DMatrix): [number, number] { //todo maybe merge with  maxfunction ?
    let max = -Infinity;
    let min = Infinity;
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr[i].length; j++) {
            if (arr[i][j]) {
                //console.log("length : ", arr[i][j].length);
                for (let k = 0; k < arr[i][j].length; k++) {
                    if (arr[i][j][k] < min) {
                        min = arr[i][j][k];
                    }
                    if (arr[i][j][k] > max) {
                        max = arr[i][j][k];
                    }
                }
            }
        }
    }
    return [min, max];
}

export function convertNumberToStringPercentage(value: number): string {
    return (value * 100).toFixed(2) + "%";
}

function projectTo255(x: number, maxVisualizationInput: number): number { //todo remove the other 255 func
    return ((x + maxVisualizationInput) / (2 * maxVisualizationInput)) * 255;
}

function mapTo255(minValue: number, maxValue: number, value: number): number {
    return Math.round((value - minValue) / (maxValue - minValue) * 255);
}

export function getInputVectorAsRGB2DImage(inputVector: numberVector, size: number, maxVisualizationInput: number): RGB2DImage {
    const latentVectorAsMatrix: RGB2DImage = getEmptyRGB2DImage(size, size);

    let intensityProjected: number;
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            intensityProjected = projectTo255(inputVector[i * size + j], maxVisualizationInput); //between 0 black and 255 white

            latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
        }
    }
    return latentVectorAsMatrix;
}

export function getRandomNormalFloat(mu: number, sigma: number): number {
    const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
    return mu + sigma * z;
}
