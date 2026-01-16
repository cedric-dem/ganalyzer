type MatrixCell = [number, number, number];
type Matrix2D = Array<Array<MatrixCell | null>>;
type NestedNumberArray = number | NestedNumberArray[];

function getUpperBoundSqrt(n: number): number {
    return Math.ceil(Math.sqrt(n));
}

export function get2DNullArray(sizeX: number, sizeY: number): Matrix2D {
    return Array.from({length: sizeX}, () =>
        Array.from({length: sizeY}, () => null)
    );
}

export function get1DNullArray(size: number): number[] {
    return Array(size).fill(0);
}

function getResultFrom1DCase(content: any, min: number, max: number): Matrix2D {
    const newDimension = getUpperBoundSqrt(content.length);

    const result = get2DNullArray(newDimension, newDimension);

    let tempValue: number;
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

function getResultFrom3DCase(content: any, minimum: number, maximum: number): Matrix2D {
    //todo maybe use three js here ?
    //or slider to have several 2d pictures and allowed to go trough the frames
    const outerDimension = getUpperBoundSqrt(content[0][0].length);
    const innerDimension = content.length;
    const margin = Math.ceil(10 / outerDimension); //margin depends on the number of "smaller images"

    const result = get2DNullArray(outerDimension * (innerDimension + margin), outerDimension * (innerDimension + margin));

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

export function getMatrixToDisplay(rawContent: number[] | number[][][]): Matrix2D | undefined {

    const dimensionsQuantity = getArrayDimensions(rawContent);

    const minimum = getOverallMinimum(rawContent);
    const maximum = getOverallMaximum(rawContent);

    let result: Matrix2D | undefined;

    if (dimensionsQuantity === 1) {
        result = getResultFrom1DCase(rawContent, minimum, maximum);

    } else if (dimensionsQuantity === 3) {
        result = getResultFrom3DCase(rawContent, minimum, maximum);

    } else {
        console.log('ERROR');
    }

    return result;
}

function getArrayDimensions(array: unknown): number {
    let dimensionsQuantity = 0;
    let cursor = array;

    while (Array.isArray(cursor)) {
        dimensionsQuantity++;
        cursor = cursor[0];
    }

    return dimensionsQuantity;
}

function getOverallMinimum(array: NestedNumberArray[]): number {
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

function getOverallMaximum(array: NestedNumberArray[]): number {
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

export function toPercentage(value: number): string {
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

export function getInputVectorAsMatrix(inputVector: number[], size: number, maxVisualizationInput: number): Matrix2D {
    const latentVectorAsMatrix = get2DNullArray(size, size);

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const intensityProjected = projectTo255(inputVector[i * size + j], maxVisualizationInput); //between 0 black and 255 white

            latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
        }
    }
    return latentVectorAsMatrix;
}

export function getRandomNormalFloat(mu: number, sigma: number): number {
    const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
    return mu + sigma * z;
}