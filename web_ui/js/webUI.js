import ApiClient from "./apiClient.js";
import DiscriminatorController from "./discriminatorController.js";
import GeneratorController from "./generatorController.js";
import {ImageGridRenderer, SliderGridRenderer} from "./renderers.js";
import {addChoices} from "./misc.js";

class WebUI {
    constructor({modelName, imageSize, latentSpaceSize, maxValueVisualizationInput, apiBaseUrl}) {

        this.modelName = modelName;

        this.availableEpochs = null;

        this.imageSize = imageSize;

        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSize ** 0.5;

        this.maxValueVisualizationInput = maxValueVisualizationInput;

        this.apiClient = new ApiClient(apiBaseUrl);

        this.imageGridRenderer = new ImageGridRenderer();
        this.sliderGridRenderer = new SliderGridRenderer();

        this.discriminatorController = new DiscriminatorController(this, this.apiClient, this.imageGridRenderer);
        this.generatorController = new GeneratorController(this, this.apiClient, this.imageGridRenderer, this.sliderGridRenderer, this.discriminatorController, latentSpaceSize);
    }

    initialize() {
        this.generatorController.initialize();
        this.discriminatorController.initialize();

        this.apiClient.synchronizeServer(this.modelName, this.latentSpaceSize).then((data) => {

            const generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue");
            const discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue");

            this.availableEpochs = data.number_of_models;

            generatorEpochSlider.max = this.availableEpochs;
            discriminatorEpochSlider.max = this.availableEpochs;

            addChoices(this.generatorController, true, "choice_layer_generator", data.generator_layers);
            addChoices(this.discriminatorController, false, "choice_layer_discriminator", data.discriminator_layers);

            this.generatorController.initializeLastLayer(data.generator_layers[data.generator_layers.length -1])
            this.discriminatorController.initializeLastLayer(data.discriminator_layers[data.discriminator_layers.length -1])

            const generatorEpochValue = generatorEpochSlider.value;
            this.generatorController.updateEpoch(generatorEpochValue, false);

            const discriminatorEpochValue = discriminatorEpochSlider.value;
            this.discriminatorController.updateEpoch(discriminatorEpochValue, false);

            this.generatorController.randomizeInput();
        });

        window.handleSliderMuValue = (value) => {
            document.getElementById("sliderMuValueLabel").textContent = value;
            this.generatorController.randomizeInput();
        };

        window.handleSliderSigmaValue = (value) => {
            document.getElementById("sliderSigmaValueLabel").textContent = value;
            this.generatorController.randomizeInput();
        };

        window.handleSliderConstantValue = (value) => {
            document.getElementById("sliderConstantValueLabel").textContent = value;
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
}

export default WebUI;