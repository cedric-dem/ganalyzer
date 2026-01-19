import ApiClient from "./apiClient";
import DiscriminatorController from "./controller/discriminatorController";
import GeneratorController from "./controller/generatorController";
import InputDataController from "./controller/inputDataController";
import {number3DMatrix} from "./types/types";

type WebUIConfig = {
    modelName: string;
    imageSize: number;
    latentSpaceSize: number;
    maxValueVisualizationInput: number;
    apiBaseUrl: string;
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
    public latentSpaceSizeSqrt: number;
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
        this.generatorController = new GeneratorController(this, this.apiClient, "choice_layer_generator");

        this.generatorEpochSlider = document.getElementById("sliderGeneratorEpochValue") as HTMLInputElement | null;
        this.discriminatorEpochSlider = document.getElementById("sliderDiscriminatorEpochValue") as HTMLInputElement | null;
    }

    initialize() {
        if (!this.generatorEpochSlider || !this.discriminatorEpochSlider) {
            throw new Error("Epoch slider elements are missing from the DOM.");
        }

        this.generatorController.initialize();
        this.discriminatorController.initialize();

        this.apiClient.synchronizeServer(this.modelName, this.latentSpaceSize).then((data: any) => {
            if (!data) {
                return;
            }

            this.availableEpochs = data.number_of_models;

            this.generatorEpochSlider.max = String(this.availableEpochs);
            this.discriminatorEpochSlider.max = String(this.availableEpochs);

            this.generatorController.setLayers(data.generator_layers);
            this.discriminatorController.setLayers(data.discriminator_layers);

            this.generatorController.updateEpoch(parseFloat(this.generatorEpochSlider.value), false);
            this.discriminatorController.updateEpoch(parseFloat(this.discriminatorEpochSlider.value), false);

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
            this.generatorController.updateEpoch(parseFloat(newEpoch));
        };

        window.handleSliderDiscriminatorEpochValue = (newEpoch) => {
            this.discriminatorController.updateEpoch(parseFloat(newEpoch));
        };
    }

    async updateDiscriminator(dataGenerator: number3DMatrix) {
        this.discriminatorController.changeInputImage(dataGenerator );
        await this.discriminatorController.refreshAll();
    }

    getLatentVector(): number[] {
        return this.inputDataController.getLatentVector();
    }

    refreshGenerator() {
        this.generatorController.refreshAll();
    }
}