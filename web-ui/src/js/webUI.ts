import ApiClient from "./apiClient.js";
import DiscriminatorController from "./controller/discriminatorController.js";
import GeneratorController from "./controller/generatorController.js";
import InputDataController from "./controller/inputDataController.js";

type WebUIConfig = {
    modelName: string;
    imageSize: number;
    latentSpaceSize: number;
    maxValueVisualizationInput: number;
    apiBaseUrl: string;
};

type SyncServerResponse = {
    number_of_models: number;
    generator_layers: string[];
    discriminator_layers: string[];
};

declare global {
    interface Window {
        handleSliderMuValue?: (value: string) => void;
        handleSliderSigmaValue?: (value: string) => void;
        handleSliderConstantValue?: (value: string) => void;
        re_randomize?: () => void;
        set_constant_input?: () => void;
        handleSliderGeneratorEpochValue?: (newEpoch: string) => void;
        handleSliderDiscriminatorEpochValue?: (newEpoch: string) => void;
    }
}

export default class WebUI {
    modelName: string;
    availableEpochs: number | null;
    imageSize: number;
    latentSpaceSize: number;
    latentSpaceSizeSqrt: number;
    maxValueVisualizationInput: number;
    apiClient: ApiClient;
    inputDataController: InputDataController;
    discriminatorController: DiscriminatorController;
    generatorController: GeneratorController;
    generatorEpochSlider: HTMLInputElement | null;
    discriminatorEpochSlider: HTMLInputElement | null;

    constructor({modelName, imageSize, latentSpaceSize, maxValueVisualizationInput, apiBaseUrl}: WebUIConfig) {
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

        this.generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue") as HTMLInputElement | null;
        this.discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue") as HTMLInputElement | null;
    }

    initialize() {
        if (!this.generatorEpochSlider || !this.discriminatorEpochSlider) {
            throw new Error("Epoch slider elements are missing from the DOM.");
        }

        this.generatorController.initialize();
        this.discriminatorController.initialize();

        this.apiClient.synchronizeServer(this.modelName, this.latentSpaceSize).then((data: SyncServerResponse | null) => {
            if (!data) {
                return;
            }

            this.availableEpochs = data.number_of_models;

            this.generatorEpochSlider.max = String(this.availableEpochs);
            this.discriminatorEpochSlider.max = String(this.availableEpochs);

            this.generatorController.setLayers(data.generator_layers);
            this.discriminatorController.setLayers(data.discriminator_layers);

            this.generatorController.updateEpoch(this.generatorEpochSlider.value, false);
            this.discriminatorController.updateEpoch(this.discriminatorEpochSlider.value, false);

            this.inputDataController.randomizeInput();
        });

        window.handleSliderMuValue = (value) => {
            this.inputDataController.handleSliderMuValue(value);
        };

        window.handleSliderSigmaValue = (value) => {
            this.inputDataController.handleSliderSigmaValue(value);
        };

        window.handleSliderConstantValue = (value) => {
            this.inputDataController.handleSliderConstantValue(value);
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

    async updateDiscriminator(dataGenerator: unknown) {
        this.discriminatorController.changeInputImage(dataGenerator);
        await this.discriminatorController.refreshAll();
    }

    getLatentVector(): number[] {
        return this.inputDataController.getLatentVector();
    }

    refreshGenerator() {
        this.generatorController.refreshAll();
    }
}