import {changeInsideRepresentation} from "./misc.js";

class GeneratorController {
    constructor(calling_web_ui, apiClient, imageGridRenderer, sliderGridRenderer, discriminatorController) {
        this.callingWebUi = calling_web_ui;
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
            "div_visualization_input_generator",
            this.callingWebUi.latentSpaceSizeSqrt,
            this.callingWebUi.latentSpaceSizeSqrt,
        );
        this.generatorImagePixels = this.imageGridRenderer.initializeImage(
            "div_visualization_output_generator",
            this.callingWebUi.imageSize,
            this.callingWebUi.imageSize,
        );
        this.slidersGrid = this.sliderGridRenderer.initializeGeneratorSliders(
            "sliders_grid",
            this.callingWebUi.latentSpaceSizeSqrt,
            (i, j, newValue) => this.handleSliderValueChange(i, j, newValue),
        );
    }

    async refreshGeneratorAndDiscriminator() {
        const latentVectorAsMatrix = this.getInputVectorAsMatrix();

        this.imageGridRenderer.changeImage(latentVectorAsMatrix, this.generatorInputPixels);

        const data_generator = await this.apiClient.getModelPrediction(this.callingWebUi.latentVector, "generator", "23) conv2d");
        const data_discriminator = await this.apiClient.getModelPrediction(data_generator, "discriminator", "17) dense");

        if (!data_generator || !data_discriminator) {
            return;
        }

        this.imageGridRenderer.changeImage(data_generator, this.generatorImagePixels);

        this.refreshInsideGeneratorNew(document.getElementById("choice_layer_generator").value);

        await this.discriminatorController.refreshDiscriminator(data_generator, data_discriminator);
    }

    async refreshInsideGeneratorNew(layer_to_visualize) {
        //console.log("+++ refreshing generator inside", layer_to_visualize)

        //api call with the current layer and 'generator'
        this.getInputVectorAsMatrix();

        const inside_values_generator = await this.apiClient.getModelPrediction(this.callingWebUi.latentVector, "generator", layer_to_visualize);

        //change image
        changeInsideRepresentation(inside_values_generator, "div_visualization_inside_generator")
    }

    projectTo255(x) {
        const clamped = Math.min(Math.max(x, -this.callingWebUi.maxValueVisualizationInput), this.callingWebUi.maxValueVisualizationInput);
        return ((clamped + this.callingWebUi.maxValueVisualizationInput) / (2 * this.callingWebUi.maxValueVisualizationInput)) * 255;
    }

    getInputVectorAsMatrix() {
        const latentVectorAsMatrix = Array.from(
            {length: this.callingWebUi.latentSpaceSizeSqrt},
            () => Array(this.callingWebUi.latentSpaceSizeSqrt).fill(null),
        );

        for (let i = 0; i < this.callingWebUi.latentSpaceSizeSqrt; i++) {
            for (let j = 0; j < this.callingWebUi.latentSpaceSizeSqrt; j++) {
                const intensity = this.callingWebUi.latentVector[i * this.callingWebUi.latentSpaceSizeSqrt + j];
                const intensityProjected = this.projectTo255(intensity); //between 0 black and 255 white

                latentVectorAsMatrix[i][j] = [intensityProjected, intensityProjected, intensityProjected];
            }
        }
        return latentVectorAsMatrix;
    }

    randomizeInput() {
        const mu = parseFloat(document.getElementById("sliderMuValue").value);
        const sigma = parseFloat(document.getElementById("sliderSigmaValue").value);

        for (let i = 0; i < this.callingWebUi.latentSpaceSize; i++) {
            const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
            this.callingWebUi.latentVector[i] = mu + sigma * z;
        }
        this.refreshSliders();
        this.refreshGeneratorAndDiscriminator();
    }

    setConstantInput() {
        const k = parseFloat(document.getElementById("sliderConstantValue").value);
        for (let i = 0; i < this.callingWebUi.latentVector.length; i++) {
            this.callingWebUi.latentVector[i] = k;
        }
        this.refreshSliders();
        this.refreshGeneratorAndDiscriminator();
    }

    refreshSliders() {
        this.sliderGridRenderer.refreshSliders(this.slidersGrid, this.callingWebUi.latentVector, this.callingWebUi.latentSpaceSizeSqrt);
    }

    handleSliderValueChange(i, j, newValue) {
        this.callingWebUi.latentVector[i * this.callingWebUi.latentSpaceSizeSqrt + j] = newValue;
        this.refreshGeneratorAndDiscriminator();
    }

    async updateGeneratorEpoch(newEpoch, shouldRefresh = true) {
        //send message to python api
        const foundEpoch = await this.apiClient.changeEpoch("generator", newEpoch);

        //change text
        document.getElementById("labelGeneratorEpochValue").textContent =
            "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUi.availableEpochs;
        if (shouldRefresh) {
            this.refreshGeneratorAndDiscriminator();
        }
    }
}

export default GeneratorController;