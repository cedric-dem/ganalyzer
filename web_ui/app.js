//todo : why a generate calls generate twice

import ApiClient from "./js/apiClient.js";
import AppState from "./js/appState.js";
import CONFIG from "./js/config.js";
import DiscriminatorController from "./js/discriminatorController.js";
import GeneratorController from "./js/generatorController.js";
import {ImageGridRenderer, SliderGridRenderer} from "./js/renderers.js";
import {addChoices, updateDiscriminatorEpoch, updateGeneratorEpoch} from "./js/ui.js";

function bootstrapApp() {
    const state = new AppState(CONFIG);
    const apiClient = new ApiClient(CONFIG.apiBaseUrl);
    const imageGridRenderer = new ImageGridRenderer();
    const sliderGridRenderer = new SliderGridRenderer();
    const discriminatorController = new DiscriminatorController(state, apiClient, imageGridRenderer);
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
        addChoices(generatorController, true, "choice_layer_generator", data.generator_layers);

        //todo change list visual to data.generator_layers
        addChoices(discriminatorController, false, "choice_layer_discriminator", data.discriminator_layers);

        const generatorEpochValue = generatorEpochSlider.value;
        updateGeneratorEpoch(generatorEpochValue, state, apiClient, generatorController, false);

        const discriminatorEpochValue = discriminatorEpochSlider.value;
        updateDiscriminatorEpoch(discriminatorController, discriminatorEpochValue, state, apiClient);

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
        updateDiscriminatorEpoch(discriminatorController, newEpoch, state, apiClient);
    };
}


bootstrapApp();
