import {changeInsideRepresentation, getInputVectorAsMatrix} from "./misc.js";

class GeneratorController {
    constructor(callingWebUI, apiClient, imageGridRenderer, sliderGridRenderer, discriminatorController) {
        this.callingWebUi = callingWebUI;
        this.apiClient = apiClient;
        this.imageGridRenderer = imageGridRenderer;

        this.sliderGridRenderer = sliderGridRenderer;

        this.discriminatorController = discriminatorController;

        this.generatorInputPixels = null;
        this.generatorImagePixels = null;

        this.slidersGrid = null;

        this.lastLayerName = null;
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

    initializeLastLayer(lastLayerName){
        this.lastLayerName = lastLayerName
    }

    async refreshGeneratorAndDiscriminator() {
        // input
        const latentVectorAsMatrix = getInputVectorAsMatrix(this.callingWebUi.latentVector, this.callingWebUi.latentSpaceSizeSqrt, this.callingWebUi.maxValueVisualizationInput);
        this.imageGridRenderer.changeImage(latentVectorAsMatrix, this.generatorInputPixels);

        //inside
        await this.refreshInsideGenerator(document.getElementById("choice_layer_generator").value);

        //output
        const dataGenerator = await this.apiClient.getModelPrediction(this.callingWebUi.latentVector, "generator", this.lastLayerName);
        this.imageGridRenderer.changeImage(dataGenerator, this.generatorImagePixels);

        // update discriminator
        this.discriminatorController.changeInputImage(dataGenerator);
        await this.discriminatorController.refreshDiscriminator();
    }

    async refreshInsideGenerator(layerToVisualize) {

        //api call with the current layer and 'generator'
        const insideValuesGenerator = await this.apiClient.getModelPrediction(this.callingWebUi.latentVector,"generator", layerToVisualize);

        //change image
        changeInsideRepresentation(insideValuesGenerator, "div_visualization_inside_generator")
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
        document.getElementById("labelGeneratorEpochValue").textContent = "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUi.availableEpochs;

        if (shouldRefresh) {
            await this.refreshGeneratorAndDiscriminator();
        }
    }
}

export default GeneratorController;