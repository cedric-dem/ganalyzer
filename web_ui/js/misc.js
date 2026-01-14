import {ImageGridRenderer} from "./renderers.js";

export function changeInsideRepresentation(content, location) {
    //empty previous content if any
    document.getElementById(location).innerHTML = "";

    // get matrix
    const matrix_ready_to_display = get_matrix_to_display(content);

    // create pixels
    const imageGridRenderer = new ImageGridRenderer()
    const locationImage = imageGridRenderer.initializeImage(location, matrix_ready_to_display.length, matrix_ready_to_display[0].length);

    // colour pixels
    imageGridRenderer.changeImage(matrix_ready_to_display, locationImage);
}

function getUpperBoundSqrt(n) {
    return Math.ceil(Math.sqrt(n));
}

function getDefaultMatrix(sizeX, sizeY) {
    return Array.from({length: sizeX}, () =>
        Array.from({length: sizeY}, () => null)
    );
}

function getResultFrom1DCase(content, min, max) {
    const new_dim = getUpperBoundSqrt(content.length)

    const result = getDefaultMatrix(new_dim, new_dim)

    let tmp;
    for (let i = 0; i < new_dim; i++) {
        for (let j = 0; j < new_dim; j++) {
            if (i * new_dim + j < content.length) {
                tmp = mapTo255(min, max, content[i * new_dim + j]);
                result[i][j] = [tmp, tmp, tmp];
            }
        }
    }

    return result;
}

function getResultFrom3DCase(content, min, max) {
    //todo maybe use three js here ?
    //or slider to have several 2d pictures and allowed to go trough the frames
    const outer_dimension = getUpperBoundSqrt(content[0][0].length)
    const inner_dimension = content.length

    const result = getDefaultMatrix(outer_dimension * inner_dimension, outer_dimension * inner_dimension);

    let tmp;

    for (let outer_x = 0; outer_x < outer_dimension; outer_x++) {
        for (let outer_y = 0; outer_y < outer_dimension; outer_y++) {

            for (let inner_x = 0; inner_x < inner_dimension; inner_x++) {
                for (let inner_y = 0; inner_y < inner_dimension; inner_y++) {

                    if (outer_x * outer_dimension + outer_y < content[0][0].length) {
                        // 255 white --- 0 black
                        tmp = mapTo255(min, max, content[inner_x][inner_y][outer_x * outer_dimension + outer_y]);
                        result[outer_x * inner_dimension + inner_x][outer_y * inner_dimension + inner_y] = [tmp, tmp, tmp]
                    }
                }
            }
        }
    }

    return result;
}

export function get_matrix_to_display(content) {

    const dims = getArrayDimensions(content);

    const min = getOverallMinimum(content);
    const max = getOverallMaximum(content);

    let result;

    if (dims === 1) {
        result = getResultFrom1DCase(content, min, max);

    } else if (dims === 3) {
        result = getResultFrom3DCase(content, min, max);

    } else {
        console.log('ERROR')
    }

    return result
}

function getArrayDimensions(arr) {
    let dimensions = 0;

    while (Array.isArray(arr)) {
        dimensions++;
        arr = arr[0];
    }

    return dimensions;
}

function getOverallMinimum(mat) {
    let min = Infinity;
    for (const element of mat) {
        if (Array.isArray(element)) {
            min = Math.min(min, getOverallMinimum(element));
        } else {
            min = Math.min(min, element);
        }
    }
    return min;
}

function getOverallMaximum(mat) {
    let max = -Infinity;
    for (const element of mat) {
        if (Array.isArray(element)) {
            max = Math.max(max, getOverallMaximum(element));
        } else {
            max = Math.max(max, element);
        }
    }
    return max;
}

export function addChoices(controller, isGenerator, location, layersList) { //todo coulr remove boolean isgenerator
    const select = document.getElementById(location);

    layersList.forEach((layer) => {
        const option = new Option(layer);
        select.add(option);
    });

    select.addEventListener("change", () => {
        const value = select.value;

        if (isGenerator) { //todo refactor code, both controller should inherit from controler abstract class so no if needed here
            controller.refreshInsideGenerator(value)
        } else {
            controller.refreshInsideDiscriminator(value)
        }
    });
}

export function toPercentage(value) {
    return (value * 100).toFixed(2) + "%";
}

function projectTo255(x, max_visualization_input) { //todo remove the other 255 func
    const clampedVal = Math.min(Math.max(x, -max_visualization_input), max_visualization_input);
    return ((clampedVal + max_visualization_input) / (2 * max_visualization_input)) * 255;
}

function mapTo255(min_val, max_val, val) {
    const clampedVal = Math.min(Math.max(val, min_val), max_val);
    const ratio = (clampedVal - min_val) / (max_val - min_val);
    return Math.round(ratio * 255);
}

export function getInputVectorAsMatrix(input_vector, size, max_visualization_input) {
    const latentVectorAsMatrix = getDefaultMatrix(size, size)

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const intensity = input_vector[i * size + j];
            const intensityProjected = projectTo255(intensity, max_visualization_input); //between 0 black and 255 white

            latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
        }
    }
    return latentVectorAsMatrix;
}
