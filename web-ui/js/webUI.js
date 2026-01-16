import ApiClient from "./apiClient.js";
import DiscriminatorController from "./controller/discriminatorController.js";
import GeneratorController from "./controller/generatorController.js";
import InputDataController from "./controller/inputDataController.js";

export default class WebUI {
    constructor({modelName, imageSize, latentSpaceSize, maxValueVisualizationInput, apiBaseUrl}) {

        this.modelName = modelName;

        this.availableEpochs = null;

        this.imageSize = imageSize;

        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSize ** 0.5;

        this.maxValueVisualizationInput = maxValueVisualizationInput;

        this.apiClient = new ApiClient(apiBaseUrl);

        this.inputDataController = new InputDataController(this, this.latentSpaceSize, this.latentSpaceSizeSqrt);

        this.discriminatorController = new DiscriminatorController(this, this.apiClient, "choice_layer_discriminator");
        this.generatorController = new GeneratorController(this, this.apiClient, latentSpaceSize, "choice_layer_generator");

        this.generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue");
        this.discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue");
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

            this.inputDataController.randomizeInput();
        });

        window.handleSliderMuValue = (value) => {
            this.inputDataController.handleSliderMuValue(value)
        };

        window.handleSliderSigmaValue = (value) => {
            this.inputDataController.handleSliderSigmaValue(value)
        };

        window.handleSliderConstantValue = (value) => {
            this.inputDataController.handleSliderConstantValue(value)
        };

        window.re_randomize = () => {
            this.inputDataController.randomizeInput();
        };

        window.set_constant_input = () => {
            this.inputDataController.setConstantInput();
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

    getLatentVector() {
        return this.inputDataController.getLatentVector();
    }

    refreshGenerator() {
        this.generatorController.refreshAll()
    }
}