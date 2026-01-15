import {getMatrixToDisplay} from "./misc.js";
import {ImageGridRenderer} from "./imageRenderer.js";

export class ModelController {
    constructor(callingWebUI, modelName, apiClient, locationInsideVisualization, locationEpochLabel) {

        this.callingWebUI = callingWebUI;
        this.modelName = modelName;
        this.apiClient = apiClient;

        this.inputData = null;

        this.locationEpochLabel = document.getElementById(locationEpochLabel)

        this.rendererInside = new ImageGridRenderer(locationInsideVisualization);

    }

    async refreshAll() {

    }

    async refreshInside(layerToVisualize) {

        //api call with the current layer and 'discriminator'
        const newInsideValues = await this.apiClient.getModelPrediction(this.inputData, this.modelName, layerToVisualize);

        //change image
        this.changeInsideRepresentation(newInsideValues)

    }

    changeInsideRepresentation(content) {
        //empty previous content if any
        //document.getElementById(this.locationInsideVisualization).innerHTML = "";

        // get matrix
        const matrixReadyToDisplay = getMatrixToDisplay(content);

        // create pixels
        this.rendererInside.initializeImage( matrixReadyToDisplay.length, matrixReadyToDisplay[0].length);

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
        this.locationEpochLabel.textContent = "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUI.availableEpochs;

        if (shouldRefresh) {
            await this.refreshAll();
        }
    }

    addChoices(location, layersList) {
        const select = document.getElementById(location);

        layersList.forEach((layer) => {
            const option = new Option(layer);
            select.add(option);
        });

        select.addEventListener("change", () => {
            this.refreshInside(select.value)
        });
    }

}