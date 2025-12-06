//todo : why a generate calls generate twice

async function synchronize_server() {
    //todo introduce the following settings in the gui,
    // like on the menu page the user can select the model and ls
    //send model size, ls,
    // could also send image size but i only trained on 100
    // retrieve list of layers in  both models (maybe should be in different function ?), AVAILABLE_EPOCHS also ?


    try {
        const response = await fetch("http://127.0.0.1:5000/sync-server", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model_size: MODEL_NAME,
                latent_space_size: LATENT_SPACE_SIZE
            }),
        });

        const data = await response.json();

        //todo change list visual to data.discriminator_layers
        add_choices(true, "choice_layer_generator", data.generator_layers)

        //todo change list visual to data.generator_layers
        add_choices(false, "choice_layer_discriminator", data.discriminator_layers)

    } catch (error) {
        console.error("Error:", error);
    }

}

function add_choices(is_generator, location, layers_list) {
    const select = document.getElementById(location);

    var temp_option;
    layers_list.forEach(layer => {
        temp_option = new Option(layer)
        select.add(temp_option)
    })

    var value;
    select.addEventListener("change", () => {
        value = select.value;
        change_inside_visualisation(is_generator, value)
    })

}

function change_inside_visualisation(is_generator, selected_value) {
    console.log("change inside visualization ", is_generator, selected_value)

    var values_to_display;
    var location;

    if (is_generator) {
        values_to_display = input_values_generator[selected_value]
        location = "grid_visual_inside_discriminator"

    } else {
        values_to_display = input_values_discriminator[selected_value]
        location = "grid_visual_inside_generator"
    }

    console.log(">> values to add in input field", values_to_display, "location : ", location)


    // TODO empty location
    // TODO initialize div
    // TODO fill with all_values_list[selected_value]

}

async function get_result_generator() {

    const latent_vector_as_matrix = get_input_vector_as_matrix()
    change_image(latent_vector_as_matrix, generator_input_pixels)

    try {
        const response = await fetch("http://127.0.0.1:5000/get-result-generator", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({vector: generator_input}),
        });

        const data = await response.json();
        change_image(data.generated_image, generator_image_pixels)
        refresh_inside_generator()

        //todo modify
        input_values_generator = {
            "input": [23, 24],
            "gen1": [4, 4, 45],
            "gen2": [879, 7, 4, 5],
            "gen3": [41, 2],
            "out": [4]
        }
        input_values_discriminator = {
            "input": [23, 214],
            "disc1": [44, 41, 145],
            "disc2": [879, 7, 41, 5],
            "disc3": [41, 12],
            "out": [14]
        }

        refresh_discriminator(data.generated_image, data.result_discriminator)


    } catch (error) {
        console.error("Error:", error);
    }
}

function refresh_inside_generator() {
    current_value = document.getElementById("choice_layer_generator").value
    //change_inside_visualisation(true, current_value) //todo why cant be uncommented
}

function refresh_inside_discriminator() {
    current_value = document.getElementById("choice_layer_discriminator").value
    change_inside_visualisation(false, current_value)
}


function toPercentage(value) {
    return (value * 100).toFixed(2) + "%";
}

function refresh_discriminator(new_image, result_discriminator) {
    //change input
    change_image(new_image, discriminator_input_image_pixels)

    // print it
    text_result = ""

    if (result_discriminator > 0.5) {
        text_result = "real image"
    } else {
        text_result = "fake image"
    }
    text_result += " (" + toPercentage(result_discriminator) + ")"

    document.getElementById("prediction_output_text").textContent = text_result;

    refresh_inside_discriminator()
}

async function change_epoch(model_type, new_epoch) {
    // console.log('change epoch' + model_type);
    try {
        const response = await fetch("http://127.0.0.1:5000/change-epoch-" + model_type, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({new_epoch: new_epoch}),
        });

        const data = await response.json();
        // todo add error if not changed
        return data.new_epoch_found

    } catch (error) {
        console.error("Error:", error);
    }
}


function get_input_vector_as_matrix() {
    var latent_vector_as_matrix = Array.from({length: LATENT_SPACE_SIZE_SQRT}, () => Array(LATENT_SPACE_SIZE_SQRT).fill(null));

    for (var i = 0; i < LATENT_SPACE_SIZE_SQRT; i++) {
        for (var j = 0; j < LATENT_SPACE_SIZE_SQRT; j++) {
            intensity = generator_input[i * LATENT_SPACE_SIZE_SQRT + j]
            intensity_projected = projectTo255(intensity)//between 0 black and 255 white
            latent_vector_as_matrix[i][j] = [intensity_projected, intensity_projected, intensity_projected]
        }
    }
    return latent_vector_as_matrix;
}

function projectTo255(x) {
    x = Math.min(Math.max(x, -MAX_VALUE_VISUALIZATION_INPUT), MAX_VALUE_VISUALIZATION_INPUT);
    return ((x + MAX_VALUE_VISUALIZATION_INPUT) / (2 * MAX_VALUE_VISUALIZATION_INPUT)) * 255;
}

function randomize_input() {
    const mu = parseFloat(document.getElementById("sliderMuValue").value);
    const sigma = parseFloat(document.getElementById("sliderSigmaValue").value);

    for (let i = 0; i < LATENT_SPACE_SIZE; i++) {
        const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
        generator_input[i] = mu + sigma * z;
    }
    refresh_sliders()
    get_result_generator()
}

function set_constant_input() {
    const k = parseFloat(document.getElementById("sliderConstantValue").value);
    for (let i = 0; i < generator_input.length; i++) {
        generator_input[i] = k;
    }
    refresh_sliders()
    get_result_generator()
}

function re_randomize() {
    randomize_input()
    get_result_generator()
}


function refresh_sliders() {
    for (let i = 0; i < LATENT_SPACE_SIZE_SQRT; i++) {
        for (let j = 0; j < LATENT_SPACE_SIZE_SQRT; j++) {
            sliders_grid[i][j].value = generator_input[i * 7 + j];
        }
    }
}

function change_image(new_data, location) {
    for (let i = 0; i < new_data.length; i++) {
        for (let j = 0; j < new_data[0].length; j++) {
            const [r, g, b] = new_data[i][j];
            location[i][j].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        }
    }
}

function initialize_image(element_id, location_image, size_x, size_y) {
    const div_grid = document.getElementById(element_id);


    const availableWidth = div_grid.clientWidth;
    const availableHeight = div_grid.clientHeight;

    const pixelSize = Math.floor(Math.min(availableWidth / size_y, availableHeight / size_x));

    div_grid.style.display = 'grid';
    div_grid.style.gridTemplateColumns = `repeat(${size_y}, ${pixelSize}px)`;
    div_grid.style.gridTemplateRows = `repeat(${size_x}, ${pixelSize}px)`;


    for (let i = 0; i < size_x; i++) {
        for (let j = 0; j < size_y; j++) {
            const new_element = document.createElement('div');
            new_element.classList.add('slider_input_representation');
            div_grid.appendChild(new_element);
            new_element.style.width = `${pixelSize}px`;
            new_element.style.height = `${pixelSize}px`;

            location_image[i][j] = new_element;
        }
    }
}

function handleSliderMuValue(value) {
    document.getElementById("sliderMuValueLabel").textContent = value;
    randomize_input()
}

function handleSliderSigmaValue(value) {
    document.getElementById("sliderSigmaValueLabel").textContent = value;
    randomize_input()
}

function handleSliderConstantValue(value) {
    document.getElementById("sliderConstantValueLabel").textContent = value;
    set_constant_input()
}

function initialize_generator_image() {
    initialize_image('grid_input_generator', generator_input_pixels, LATENT_SPACE_SIZE_SQRT, LATENT_SPACE_SIZE_SQRT)
    initialize_image('grid_visual_generator', generator_image_pixels, IMAGE_SIZE, IMAGE_SIZE)
}

function initialize_discriminator_image() {
    initialize_image('grid_input_discriminator', discriminator_input_image_pixels, IMAGE_SIZE, IMAGE_SIZE)
}

/*
function initialize_inside_generator(){
    // location : grid_visual_inside_generator

}

function initialize_inside_discriminator(){
    // location : grid_visual_inside_discriminator

}
*/

function initialize_generator_sliders() {

    const div_sliders = document.getElementById("sliders_grid");

    div_sliders.style.gridTemplateColumns = `repeat(${LATENT_SPACE_SIZE_SQRT}, minmax(0, 1fr))`;
    div_sliders.style.gridTemplateRows = `repeat(${LATENT_SPACE_SIZE_SQRT}, auto)`;

    for (let i = 0; i < LATENT_SPACE_SIZE_SQRT; i++) {
        for (let j = 0; j < LATENT_SPACE_SIZE_SQRT; j++) {
            const new_element = document.createElement('input');

            new_element.type = "range";
            new_element.min = -5;
            new_element.max = 5;
            new_element.step = 0.01;
            new_element.value = 0;
            new_element.classList.add('slider');

            new_element.oninput = function () {
                handleSliderValueChange(i, j, new_element.value);
            };

            div_sliders.appendChild(new_element);
            sliders_grid[i][j] = new_element;
        }
    }
}

function handleSliderValueChange(i, j, new_value) {
    generator_input[i * LATENT_SPACE_SIZE_SQRT + j] = new_value;
    get_result_generator()
}

async function handleSliderGeneratorEpochValue() {
    new_epoch = document.getElementById("sliderGeneratorEpochValue").value;

    //send message to python api
    found_epoch = await change_epoch("generator", new_epoch)

    //change text
    document.getElementById("labelGeneratorEpochValue").textContent = "Epoch : " + new_epoch + "(" + found_epoch + ")" + "/" + AVAILABLE_EPOCHS;
    get_result_generator()

}

async function handleSliderDiscriminatorEpochValue() {
    new_epoch = document.getElementById("sliderDiscriminatorEpochValue").value;

    //send message to python api
    found_epoch = await change_epoch("generator", new_epoch)

    //change text
    document.getElementById("labelDiscriminatorEpochValue").textContent = "Epoch : " + new_epoch + "(" + found_epoch + ")" + "/" + AVAILABLE_EPOCHS;
    //get_result_discriminator()
}

/// config
const MODEL_NAME = "model_0_small"
const AVAILABLE_EPOCHS = 100;
const IMAGE_SIZE = 100;
const LATENT_SPACE_SIZE = 121;
const LATENT_SPACE_SIZE_SQRT = LATENT_SPACE_SIZE ** 0.5;

/// generator panel
let sliders_grid = Array.from({length: LATENT_SPACE_SIZE_SQRT}, () => Array(LATENT_SPACE_SIZE_SQRT).fill(null));
let generator_input = new Array(LATENT_SPACE_SIZE).fill(0);
let input_values_generator = null;

/// generator visualization
const MAX_VALUE_VISUALIZATION_INPUT = 5;
const generator_input_pixels = Array.from({length: LATENT_SPACE_SIZE_SQRT}, () => Array(LATENT_SPACE_SIZE_SQRT).fill(null));
const generator_image_pixels = Array.from({length: IMAGE_SIZE}, () => Array(IMAGE_SIZE).fill(null));

/// Discriminator
const discriminator_input_image_pixels = Array.from({length: IMAGE_SIZE}, () => Array(IMAGE_SIZE).fill(null));
let input_values_discriminator = null;

/// generator visualization

/// initialize page
initialize_generator_image()
initialize_generator_sliders()
initialize_discriminator_image()

synchronize_server().then(r =>
    randomize_input()
)

handleSliderGeneratorEpochValue()
handleSliderDiscriminatorEpochValue()