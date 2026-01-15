import ApiClient from "./apiClient.js";
import DiscriminatorController from "./controller/discriminatorController.js";
import GeneratorController from "./controller/generatorController.js";
import {SliderRenderer} from "./renderer/sliderRenderer.js";

class WebUI {
    constructor({modelName, imageSize, latentSpaceSize, maxValueVisualizationInput, apiBaseUrl}) {

        this.modelName = modelName;

        this.availableEpochs = null;

        this.imageSize = imageSize;

        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSize ** 0.5;

        this.maxValueVisualizationInput = maxValueVisualizationInput;

        this.apiClient = new ApiClient(apiBaseUrl);

        this.sliderGridRenderer = new SliderRenderer("sliders_grid");

        this.discriminatorController = new DiscriminatorController(this, this.apiClient, "choice_layer_discriminator");
        this.generatorController = new GeneratorController(this, this.apiClient, this.sliderGridRenderer, latentSpaceSize, "choice_layer_generator");

        this.generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue");
        this.discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue");

        this.sliderMuTextValue = document.getElementById("sliderMuValueLabel")
        this.sliderSigmaTextValue = document.getElementById("sliderSigmaValueLabel")
        this.sliderConstantTextValue = document.getElementById("sliderConstantValueLabel")

        this.sliderMuValue = document.getElementById("sliderMuValue")
        this.sliderSigmaValue = document.getElementById("sliderSigmaValue")
        this.sliderConstantValue = document.getElementById("sliderConstantValue")
    }

    initialize() {
        this.generatorController.initialize();
        this.discriminatorController.initialize();

        this.apiClient.synchronizeServer(this.modelName, this.latentSpaceSize).then((data) => {

            this.availableEpochs = data.number_of_models;

            this.generatorEpochSlider.max = this.availableEpochs;
            this.discriminatorEpochSlider.max = this.availableEpochs;

            this.generatorController.setLayers(data.generator_layers)
            this.discriminatorController.setLayers(data.discriminator_layers)

            this.generatorController.updateEpoch(this.generatorEpochSlider.value, false);
            this.discriminatorController.updateEpoch(this.discriminatorEpochSlider.value, false);

            this.generatorController.randomizeInput();
        });

        window.handleSliderMuValue = (value) => {
            this.sliderMuTextValue.textContent = value;
            this.generatorController.randomizeInput();
        };

        window.handleSliderSigmaValue = (value) => {
            this.sliderSigmaTextValue.textContent = value;
            this.generatorController.randomizeInput();
        };

        window.handleSliderConstantValue = (value) => {
            this.sliderConstantTextValue.textContent = value;
            this.generatorController.setConstantInput();
        };

        window.re_randomize = () => {
            this.generatorController.randomizeInput();
        };

        window.set_constant_input = () => {
            this.generatorController.setConstantInput();
        };

        window.handleSliderGeneratorEpochValue = (newEpoch) => {
            this.generatorController.updateEpoch(newEpoch);
        };

        window.handleSliderDiscriminatorEpochValue = (newEpoch) => {
            this.discriminatorController.updateEpoch(newEpoch);
        };
    }

    async updateDiscriminator(dataGenerator) {
        this.discriminatorController.changeInputImage(dataGenerator);
        await this.discriminatorController.refreshAll();
    }

    getMuValue(){
        return parseFloat(this.sliderMuValue.value)
    }

    getSigmaValue(){
        return parseFloat(this.sliderSigmaValue.value)
    }

    getKValue(){
        return parseFloat(this.sliderConstantValue.value);
    }
}

export default WebUI;