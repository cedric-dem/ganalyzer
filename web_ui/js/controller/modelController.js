import {getMatrixToDisplay} from "../misc.js";
import {ImageRenderer} from "../renderer/imageRenderer.js";

export class ModelController {
    constructor(callingWebUI, modelName, apiClient, locationInsideVisualization, locationEpochLabel, layerLocation) {

        this.callingWebUI = callingWebUI;
        this.modelName = modelName;
        this.apiClient = apiClient;

        this.inputData = null;

        this.locationEpochLabel = document.getElementById(locationEpochLabel)

        this.rendererInside = new ImageRenderer(locationInsideVisualization);

        this.selectLayerLocation = document.getElementById(layerLocation)
    }

    async refreshAll() {}

    async refreshInside(layerToVisualize) {

        //api call with the current layer and 'discriminator'
        const newInsideValues = await this.apiClient.getModelPrediction(this.inputData, this.modelName, layerToVisualize);

        //change image
        this.changeInsideRepresentation(newInsideValues)
    }

    changeInsideRepresentation(content) {
        // get matrix
        const matrixReadyToDisplay = getMatrixToDisplay(content);

        // create pixels
        this.rendererInside.initializeImage(matrixReadyToDisplay.length, matrixReadyToDisplay[0].length);

        // colour pixels
        this.rendererInside.changeImage(matrixReadyToDisplay);
    }


    initializeLastLayer(lastLayerName) {
        this.lastLayerName = lastLayerName
    }

    async updateEpoch(newEpoch, shouldRefresh = true) {

        //send message to python api
        const foundEpoch = await this.apiClient.changeEpoch(this.modelName, newEpoch);

        //change text
        this.locationEpochLabel.textContent = `Epoch : ${newEpoch} (${foundEpoch}) / ${this.callingWebUI.availableEpochs}`;

        if (shouldRefresh) {
            await this.refreshAll();
        }
    }

    addChoices(layersList) {

        layersList.forEach((layer) => {
            const option = new Option(layer);
            this.selectLayerLocation.add(option);
        });

        this.selectLayerLocation.addEventListener("change", () => {
            this.refreshInside(this.selectLayerLocation.value)
        });
    }

    setLayers(layersList){
        this.initializeLastLayer(layersList[layersList.length - 1])

        this.addChoices(layersList)

    }
}