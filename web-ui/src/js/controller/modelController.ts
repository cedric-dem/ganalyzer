import {getRGB2DImageFromRawContent} from "../misc";
import {ImageRenderer} from "../renderer/imageRenderer";
import WebUI from "../webUI";
import ApiClient from "../apiClient";
import {number3DMatrix, RGB2DImage} from "../types/types";


export class ModelController {
    protected callingWebUI: WebUI;
    protected modelName: string;
    protected apiClient: ApiClient;
    protected inputData: number3DMatrix;
    protected locationEpochLabel: HTMLElement;
    protected rendererInside: ImageRenderer;
    protected selectLayerLocation: HTMLSelectElement;
    protected lastLayerName: string;

    constructor(
        callingWebUI: WebUI,
        modelName: string,
        apiClient: ApiClient,
        locationInsideVisualization: string,
        locationEpochLabel: string,
        layerLocation: string,
    ) {
        this.callingWebUI = callingWebUI;
        this.modelName = modelName;
        this.apiClient = apiClient;

        this.locationEpochLabel = document.getElementById(locationEpochLabel);

        this.rendererInside = new ImageRenderer(locationInsideVisualization);

        this.selectLayerLocation = document.getElementById(layerLocation) as HTMLSelectElement;
    }

    async refreshAll(): Promise<void> {
    }

    async refreshInside(layerToVisualize: string): Promise<void> {
        const newInsideValues: number3DMatrix = await this.apiClient.getModelPrediction(
            this.inputData,
            this.modelName,
            layerToVisualize,
        );

        this.changeInsideRepresentation(newInsideValues);
    }

    changeInsideRepresentation(content: number3DMatrix): void {
        const matrixReadyToDisplay: RGB2DImage = getRGB2DImageFromRawContent(content);

        this.rendererInside.initializeImage(
            matrixReadyToDisplay.length,
            matrixReadyToDisplay[0].length,
        );

        this.rendererInside.changeImage(matrixReadyToDisplay);
    }

    initializeLastLayer(lastLayerName: string): void {
        this.lastLayerName = lastLayerName;
    }

    async updateEpoch(newEpoch: number, shouldRefresh = true): Promise<void> {
        const foundEpoch: number = await this.apiClient.changeEpoch(this.modelName, newEpoch);

        this.locationEpochLabel.textContent = `Epoch : ${newEpoch} (${foundEpoch}) / ${this.callingWebUI.availableEpochs}`;

        if (shouldRefresh) {
            await this.refreshAll();
        }
    }

    addChoices(layersList: string[]): void {
        layersList.forEach((layer) => {
            const option: HTMLOptionElement = new Option(layer);
            this.selectLayerLocation.add(option);
        });

        this.selectLayerLocation.addEventListener("change", () => {
            void this.refreshInside(this.selectLayerLocation.value);
        });
    }

    setLayers(layersList: string[]): void {
        this.initializeLastLayer(layersList[layersList.length - 1]);

        this.addChoices(layersList);
    }
}