//todo : why a generate calls generate twice

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async synchronizeServer(modelName, latentSpaceSize) {
        try {
            const response = await fetch(`${this.baseUrl}/sync-server`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    model_size: modelName,
                    latent_space_size: latentSpaceSize,
                }),
            });

            return await response.json();
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async getResultGenerator(vector) {
        try {
            const response = await fetch(`${this.baseUrl}/get-result-generator`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({vector: vector}),
            });

            return await response.json();
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async changeEpoch(modelType, newEpoch) {
        try {
            const response = await fetch(`${this.baseUrl}/change-epoch-${modelType}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({new_epoch: newEpoch}),
            });

            const data = await response.json();
            // todo add error if not changed
            return data.new_epoch_found;
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }
}

class AppState {
    constructor({
                    modelName,
                    availableEpochs,
                    imageSize,
                    latentSpaceSize,
                    maxValueVisualizationInput,
                }) {
        this.modelName = modelName;
        this.availableEpochs = availableEpochs;
        this.imageSize = imageSize;
        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSize ** 0.5;
        this.maxValueVisualizationInput = maxValueVisualizationInput;
        this.latentVector = new Array(latentSpaceSize).fill(0);
        this.inputValuesGenerator = null;
        this.inputValuesDiscriminator = null;
    }
}

class ImageGridRenderer {
    initializeImage(elementId, sizeX, sizeY) {
        const divGrid = document.getElementById(elementId);

        const availableWidth = divGrid.clientWidth;
        const availableHeight = divGrid.clientHeight;

        const pixelSize = Math.floor(Math.min(availableWidth / sizeY, availableHeight / sizeX));

        divGrid.style.display = "grid";
        divGrid.style.gridTemplateColumns = `repeat(${sizeY}, ${pixelSize}px)`;
        divGrid.style.gridTemplateRows = `repeat(${sizeX}, ${pixelSize}px)`;

        const locationImage = Array.from({length: sizeX}, () => Array(sizeY).fill(null));

        for (let i = 0; i < sizeX; i++) {
            for (let j = 0; j < sizeY; j++) {
                const newElement = document.createElement("div");
                newElement.classList.add("slider_input_representation");
                divGrid.appendChild(newElement);
                newElement.style.width = `${pixelSize}px`;
                newElement.style.height = `${pixelSize}px`;

                locationImage[i][j] = newElement;
            }
        }

        return locationImage;
    }

    changeImage(newData, location) {
        for (let i = 0; i < newData.length; i++) {
            for (let j = 0; j < newData[0].length; j++) {
                const [r, g, b] = newData[i][j];
                location[i][j].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
            }
        }
    }
}

class SliderGridRenderer {
    initializeGeneratorSliders(elementId, size, onInput) {
        const divSliders = document.getElementById(elementId);

        divSliders.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        divSliders.style.gridTemplateRows = `repeat(${size}, auto)`;

        const slidersGrid = Array.from({length: size}, () => Array(size).fill(null));

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const newElement = document.createElement("input");

                newElement.type = "range";
                newElement.min = -5;
                newElement.max = 5;
                newElement.step = 0.01;
                newElement.value = 0;
                newElement.classList.add("slider");

                newElement.oninput = function () {
                    onInput(i, j, newElement.value);
                };

                divSliders.appendChild(newElement);
                slidersGrid[i][j] = newElement;
            }
        }

        return slidersGrid;
    }

    refreshSliders(slidersGrid, latentVector, size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                slidersGrid[i][j].value = latentVector[i * size + j];
            }
        }
    }
}

class DiscriminatorController {
    constructor(state, imageGridRenderer) {
        this.state = state;
        this.imageGridRenderer = imageGridRenderer;
        this.discriminatorInputImagePixels = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage(
            "grid_input_discriminator",
            this.state.imageSize,
            this.state.imageSize,
        );
    }

    changeInsideVisualization(isGenerator, selectedValue) {
        console.log("change inside visualization ", isGenerator, selectedValue);

        let valuesToDisplay;
        let location;

        if (isGenerator) {
            valuesToDisplay = this.state.inputValuesGenerator?.[selectedValue];
            location = "grid_visual_inside_discriminator";
        } else {
            valuesToDisplay = this.state.inputValuesDiscriminator?.[selectedValue];
            location = "grid_visual_inside_generator";
        }

        console.log(" >> values to add in input field",
            valuesToDisplay,
            "location : ",
            location,
        );

        // TODO empty location
        // TODO initialize div
        // TODO fill with all_values_list[selected_value]
    }

    refreshInsideDiscriminator() {
        const currentValue = document.getElementById("choice_layer_discriminator").value;
        this.changeInsideVisualization(false, currentValue);
    }

    refreshDiscriminator(newImage, resultDiscriminator) {
        //change input
        this.imageGridRenderer.changeImage(newImage, this.discriminatorInputImagePixels);

        // print it
        let textResult = "";

        if (resultDiscriminator > 0.5) {
            textResult = "real image";
        } else {
            textResult = "fake image";
        }
        textResult += " (" + this.toPercentage(resultDiscriminator) + ")";

        document.getElementById("prediction_output_text").textContent = textResult;

        this.refreshInsideDiscriminator();
    }

    toPercentage(value) {
        return (value * 100).toFixed(2) + "%";
    }
}

class GeneratorController {
    constructor(state, apiClient, imageGridRenderer, sliderGridRenderer, discriminatorController) {
        this.state = state;
        this.apiClient = apiClient;
        this.imageGridRenderer = imageGridRenderer;
        this.sliderGridRenderer = sliderGridRenderer;
        this.discriminatorController = discriminatorController;
        this.generatorInputPixels = null;
        this.generatorImagePixels = null;
        this.slidersGrid = null;
    }

    initialize() {
        this.generatorInputPixels = this.imageGridRenderer.initializeImage(
            "grid_input_generator",
            this.state.latentSpaceSizeSqrt,
            this.state.latentSpaceSizeSqrt,
        );
        this.generatorImagePixels = this.imageGridRenderer.initializeImage(
            "grid_visual_generator",
            this.state.imageSize,
            this.state.imageSize,
        );
        this.slidersGrid = this.sliderGridRenderer.initializeGeneratorSliders(
            "sliders_grid",
            this.state.latentSpaceSizeSqrt,
            (i, j, newValue) => this.handleSliderValueChange(i, j, newValue),
        );
    }

    async submitToApi() {
        const latentVectorAsMatrix = this.getInputVectorAsMatrix();
        this.imageGridRenderer.changeImage(latentVectorAsMatrix, this.generatorInputPixels);

        const data = await this.apiClient.getResultGenerator(this.state.latentVector);
        if (!data) {
            return;
        }

        this.imageGridRenderer.changeImage(data.generated_image, this.generatorImagePixels);
        this.refreshInsideGenerator();

        //todo modify
        this.state.inputValuesGenerator = {
            input: [23, 24],
            gen1: [4, 4, 45],
            gen2: [879, 7, 4, 5],
            gen3: [41, 2],
            out: [4],
        };
        this.state.inputValuesDiscriminator = {
            input: [23, 214],
            disc1: [44, 41, 145],
            disc2: [879, 7, 41, 5],
            disc3: [41, 12],
            out: [14],
        };
        const resultInside = data.inside_values;
        console.log("===> inside data", resultInside);

        this.discriminatorController.refreshDiscriminator(data.generated_image, data.result_discriminator);
    }

    refreshInsideGenerator() {
        const currentValue = document.getElementById("choice_layer_generator").value;
        //this.discriminatorController.changeInsideVisualization(true, currentValue); //todo why cant be uncommented
    }

    projectTo255(x) {
        const clamped = Math.min(Math.max(x, -this.state.maxValueVisualizationInput), this.state.maxValueVisualizationInput);
        return ((clamped + this.state.maxValueVisualizationInput)
            / (2 * this.state.maxValueVisualizationInput)) * 255;
    }

    getInputVectorAsMatrix() {
        const latentVectorAsMatrix = Array.from(
            {length: this.state.latentSpaceSizeSqrt},
            () => Array(this.state.latentSpaceSizeSqrt).fill(null),
        );

        for (let i = 0; i < this.state.latentSpaceSizeSqrt; i++) {
            for (let j = 0; j < this.state.latentSpaceSizeSqrt; j++) {
                const intensity = this.state.latentVector[i * this.state.latentSpaceSizeSqrt + j];
                const intensityProjected = this.projectTo255(intensity); //between 0 black and 255 white
                latentVectorAsMatrix[i][j] = [
                    intensityProjected,
                    intensityProjected,
                    intensityProjected,
                ];
            }
        }
        return latentVectorAsMatrix;
    }

    randomizeInput() {
        const mu = parseFloat(document.getElementById("sliderMuValue").value);
        const sigma = parseFloat(document.getElementById("sliderSigmaValue").value);

        for (let i = 0; i < this.state.latentSpaceSize; i++) {
            const z = Math.sqrt(-2.0 * Math.log(Math.random()))
                * Math.cos(2.0 * Math.PI * Math.random());
            this.state.latentVector[i] = mu + sigma * z;
        }
        this.refreshSliders();
        this.submitToApi();
    }

    setConstantInput() {
        const k = parseFloat(document.getElementById("sliderConstantValue").value);
        for (let i = 0; i < this.state.latentVector.length; i++) {
            this.state.latentVector[i] = k;
        }
        this.refreshSliders();
        this.submitToApi();
    }

    reRandomize() {
        this.randomizeInput();
        this.submitToApi();
    }

    refreshSliders() {
        this.sliderGridRenderer.refreshSliders(
            this.slidersGrid,
            this.state.latentVector,
            this.state.latentSpaceSizeSqrt,
        );
    }

    handleSliderValueChange(i, j, newValue) {
        this.state.latentVector[i * this.state.latentSpaceSizeSqrt + j] = newValue;
        this.submitToApi();
    }
}

function addChoices(discriminatorController, isGenerator, location, layersList) {
    const select = document.getElementById(location);

    layersList.forEach((layer) => {
        const option = new Option(layer);
        select.add(option);
    });

    select.addEventListener("change", () => {
        const value = select.value;
        discriminatorController.changeInsideVisualization(isGenerator, value);
    });
}

const CONFIG = {
    modelName: "model_0_small",
    availableEpochs: 200,
    imageSize: 100,
    latentSpaceSize: 225,
    maxValueVisualizationInput: 5,
    apiBaseUrl: "http://127.0.0.1:5000",
};

function bootstrapApp() {
    const state = new AppState(CONFIG);
    const apiClient = new ApiClient(CONFIG.apiBaseUrl);
    const imageGridRenderer = new ImageGridRenderer();
    const sliderGridRenderer = new SliderGridRenderer();
    const discriminatorController = new DiscriminatorController(state, imageGridRenderer);
    const generatorController = new GeneratorController(
        state,
        apiClient,
        imageGridRenderer,
        sliderGridRenderer,
        discriminatorController,
    );

    generatorController.initialize();
    discriminatorController.initialize();

    const generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue");
    const discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue");

    generatorEpochSlider.max = state.availableEpochs;
    discriminatorEpochSlider.max = state.availableEpochs;

    if (Number(generatorEpochSlider.value) > state.availableEpochs) {
        generatorEpochSlider.value = state.availableEpochs;
    }
    if (Number(discriminatorEpochSlider.value) > state.availableEpochs) {
        discriminatorEpochSlider.value = state.availableEpochs;
    }

    apiClient.synchronizeServer(state.modelName, state.latentSpaceSize).then((data) => {
        if (!data) {
            return;
        }
        //todo change list visual to data.discriminator_layers
        addChoices(discriminatorController, true, "choice_layer_generator", data.generator_layers);

        //todo change list visual to data.generator_layers
        addChoices(discriminatorController, false, "choice_layer_discriminator", data.discriminator_layers);

        const generatorEpochValue = generatorEpochSlider.value;
        updateGeneratorEpoch(generatorEpochValue, state, apiClient, generatorController);

        const discriminatorEpochValue = discriminatorEpochSlider.value;
        updateDiscriminatorEpoch(discriminatorEpochValue, state, apiClient);

        generatorController.randomizeInput();
    });

    window.handleSliderMuValue = (value) => {
        document.getElementById("sliderMuValueLabel").textContent = value;
        generatorController.randomizeInput();
    };

    window.handleSliderSigmaValue = (value) => {
        document.getElementById("sliderSigmaValueLabel").textContent = value;
        generatorController.randomizeInput();
    };

    window.handleSliderConstantValue = (value) => {
        document.getElementById("sliderConstantValueLabel").textContent = value;
        generatorController.setConstantInput();
    };

    window.re_randomize = () => {
        generatorController.reRandomize();
    };

    window.set_constant_input = () => {
        generatorController.setConstantInput();
    };

    window.handleSliderGeneratorEpochValue = (newEpoch) => {
        updateGeneratorEpoch(newEpoch, state, apiClient, generatorController);
    };

    window.handleSliderDiscriminatorEpochValue = (newEpoch) => {
        updateDiscriminatorEpoch(newEpoch, state, apiClient);
    };
}

async function updateGeneratorEpoch(newEpoch, state, apiClient, generatorController) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("generator", newEpoch);
    if (foundEpoch === null) {
        return;
    }

    //change text
    document.getElementById("labelGeneratorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    generatorController.submitToApi();
}

async function updateDiscriminatorEpoch(newEpoch, state, apiClient) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("discriminator", newEpoch);
    if (foundEpoch === null) {
        return;
    }

    //change text
    document.getElementById("labelDiscriminatorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    //get_result_discriminator()
}

bootstrapApp();
