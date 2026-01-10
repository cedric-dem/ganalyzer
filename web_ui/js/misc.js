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

function getResultFrom1DCase(content) {
    const new_dim = getUpperBoundSqrt(content.length)

    const min = getOverallMinimum(content);
    const max = getOverallMaximum(content);

    //console.log('1D ', content.length, new_dim, min, max)

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

function mapTo255(min_val, max_val, val) {
    const clampedVal = Math.min(Math.max(val, min_val), max_val);
    const ratio = (clampedVal - min_val) / (max_val - min_val);
    return Math.round(ratio * 255);
}

function getResultFrom3DCase(content) {
    //todo maybe use three js here ?
    //or slider to have several 2d pictures and allowed to go trough the frames
    const outer_dimension = getUpperBoundSqrt(content[0][0].length)
    const inner_dimension = content.length


    const min = getOverallMinimum(content);
    const max = getOverallMaximum(content);

    //console.log('3D  case ====================', content[0][0].length, "outer ", outer_dimension, "inner ", inner_dimension, min, max)
    //describeMatrixShape(content)

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

    let result;

    if (dims === 1) {
        //console.log('==> 1D case')
        result = getResultFrom1DCase(content);

    } else if (dims === 3) {
        //console.log('==> 3D case')
        result = getResultFrom3DCase(content);

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

export function describeMatrixShape(new_matrix) {

    const shape = [];

    let current = new_matrix;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }

    console.log("shape ", shape, " overall min value ", getOverallMinimum(new_matrix), " overall min value ", getOverallMaximum(new_matrix));
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

export default {changeInsideRepresentation, describeMatrixShape};